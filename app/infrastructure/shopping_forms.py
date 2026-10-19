"""A2UI v0.9 自定义选购组件：SQLite 是表单及已提交需求的权威来源。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CATALOG_ID = "globex.local/shopping-v2"


class FormConflict(ValueError):
    pass


class ClarificationOption(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    value: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=200)


class ClarificationQuestion(BaseModel):
    """限制控件能力和数据大小，不预设任何选购问题或答案。"""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    type: Literal["text", "number", "single_select", "multi_select"]
    label: str = Field(min_length=1, max_length=200)
    required: bool = False
    help_text: str = Field(default="", max_length=500)
    placeholder: str = Field(default="", max_length=200)
    unit: str = Field(default="", max_length=40)
    minimum: float | None = Field(default=None, allow_inf_nan=False)
    maximum: float | None = Field(default=None, allow_inf_nan=False)
    options: list[ClarificationOption] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def check_control(self):
        if self.id in {"constructor", "prototype"}:
            raise ValueError("不支持的问题标识")
        if self.type in {"single_select", "multi_select"}:
            if not self.options or len({o.value for o in self.options}) != len(self.options):
                raise ValueError("选择题需要不重复的选项")
        elif self.options:
            raise ValueError("文本和数字问题不能带选择项")
        if self.type != "number" and (self.minimum is not None or self.maximum is not None or self.unit):
            raise ValueError("只有数字问题能设置数值范围和单位")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("最小值不能大于最大值")
        return self


class ClarificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=100, description="本次澄清的标题")
    questions: list[ClarificationQuestion] = Field(
        min_length=1, max_length=12, description="Agent 根据本轮上下文决定的问题，按展示顺序排列"
    )
    context: str = Field(default="", max_length=1000, description="已知选购背景，仅作指代，不作为买家新答案")
    description: str = Field(default="", max_length=500, description="解释本次为什么需要补充这些信息")

    @model_validator(mode="after")
    def unique_ids(self):
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("问题标识不能重复")
        return self


class PersistedForm(ClarificationRequest):
    """持久记录与当前问题定义使用同一种类型，读取时不转换历史业务字段。"""
    form_id: str
    definition_version: Literal[2]
    defaults: dict
    session_id: str
    revision: int
    origin_run_id: str
    origin_message_id: str


def validate_answers(values: dict, questions: list[dict], *, complete=False) -> dict:
    definitions = {q["id"]: ClarificationQuestion.model_validate(q) for q in questions}
    if not isinstance(values, dict) or set(values) - definitions.keys():
        raise ValueError("答案包含本次未提问的字段")
    clean = {}
    for key, question in definitions.items():
        value = values.get(key)
        if value is None or value == "" or value == [] or (isinstance(value, str) and not value.strip()):
            if complete and question.required:
                raise ValueError(f"请回答：{question.label}")
            continue
        if question.type == "text":
            if not isinstance(value, str) or len(value) > 2000:
                raise ValueError(f"{question.label}：请填写 2000 字以内的文本")
            value = value.strip()
        elif question.type == "number":
            if type(value) not in (int, float) or abs(value) > 1e15 or not math.isfinite(value):
                raise ValueError(f"{question.label}：请填写有限数值")
            if (question.minimum is not None and value < question.minimum) or (question.maximum is not None and value > question.maximum):
                raise ValueError(f"{question.label}：数值超出范围")
        else:
            choices = {option.value for option in question.options}
            if question.type == "single_select":
                if not isinstance(value, str) or value not in choices:
                    raise ValueError(f"{question.label}：请选择本次提供的选项")
            elif (not isinstance(value, list) or len(value) > len(choices)
                  or any(not isinstance(v, str) or v not in choices for v in value)
                  or len(set(value)) != len(value)):
                raise ValueError(f"{question.label}：多选答案无效")
        clean[key] = value
    return clean


def envelopes(form: dict) -> list[dict]:
    surface = form["form_id"]
    return [
        {
            "version": "v0.9",
            "createSurface": {"surfaceId": surface, "catalogId": CATALOG_ID},
        },
        {
            "version": "v0.9",
            "updateComponents": {
                "surfaceId": surface,
                "components": [
                    {
                        "id": "root",
                        "component": "ShoppingForm",
                        "title": form["title"],
                        "questions": form["questions"],
                        "context": form.get("context", ""),
                        "description": form.get("description", ""),
                        "value": {"path": "/requirements"},
                        "action": {
                            "event": {
                                "name": "applyShoppingRequirements",
                                "context": {"requirements": {"path": "/requirements"}},
                            }
                        },
                    }
                ],
            },
        },
        {
            "version": "v0.9",
            "updateDataModel": {
                "surfaceId": surface,
                "path": "/",
                "value": {
                    "requirements": form["defaults"],
                    "revision": form["revision"],
                },
            },
        },
    ]


class ShoppingFormStore:
    async def delete_session(self, buyer, session):
        def remove():
            with self._db() as db:
                db.execute("DELETE FROM shopping_forms WHERE buyer=? AND session=?",(buyer,session))
        await __import__('asyncio').to_thread(remove)
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS shopping_forms (
                id TEXT PRIMARY KEY, buyer TEXT NOT NULL, session TEXT NOT NULL,
                body TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
                submission TEXT, request_id TEXT, request_hash TEXT,
                created INTEGER NOT NULL DEFAULT (unixepoch()))""")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _public(row):
        if row is None:
            raise LookupError("选购表单不存在或不属于当前买家和会话")
        form = PersistedForm.model_validate_json(row["body"]).model_dump()
        form.update(
            created_at=row["created"] * 1000,
            revision=row["revision"],
            submission=json.loads(row["submission"]) if row["submission"] else None,
        )
        form["messages"] = envelopes(form)
        return form

    async def create_clarification(self, buyer, session, title, questions, context="", description="", *, origin_run_id="", origin_message_id=""):
        if not buyer or not session:
            raise ValueError("缺少买家会话")
        spec = ClarificationRequest.model_validate({
            "title": title, "questions": questions, "context": context, "description": description,
        })
        form = {
            "form_id": "form_" + uuid.uuid4().hex,
            "definition_version": 2,
            **spec.model_dump(),
            # 不自动补充业务问题，也不替买家预选答案。
            "defaults": {}, "session_id": session, "revision": 1,
            "origin_run_id": origin_run_id, "origin_message_id": origin_message_id,
        }
        return await self._insert(buyer, session, form)

    async def list_for_session(self, buyer, session):
        def read():
            with self._db() as db:
                rows = db.execute("SELECT * FROM shopping_forms WHERE buyer=? AND session=? ORDER BY created,rowid",
                                  (buyer,session)).fetchall()
                return [self._public(row) for row in rows]
        return await __import__('asyncio').to_thread(read)

    async def submitted_run(self, buyer, session, run_id):
        def read():
            with self._db() as db:
                row=db.execute("SELECT * FROM shopping_forms WHERE buyer=? AND session=? AND json_extract(submission,'$.run_id')=?",
                               (buyer,session,run_id)).fetchone()
                return self._public(row) if row else None
        return await __import__('asyncio').to_thread(read)

    async def _insert(self, buyer, session, form):
        def write():
            with self._db() as db:
                db.execute(
                    "INSERT INTO shopping_forms(id,buyer,session,body) VALUES(?,?,?,?)",
                    (
                        form["form_id"],
                        buyer,
                        session,
                        json.dumps(form, ensure_ascii=False),
                    ),
                )

        await asyncio.to_thread(write)
        return await self.get(buyer, session, form["form_id"])

    async def get(self, buyer, session, form_id):
        def read():
            with self._db() as db:
                return self._public(
                    db.execute(
                        "SELECT * FROM shopping_forms WHERE id=? AND buyer=? AND session=?",
                        (form_id, buyer, session),
                    ).fetchone()
                )

        return await asyncio.to_thread(read)

    async def latest_for_buyer(self, buyer):
        def read():
            with self._db() as db:
                row = db.execute(
                    "SELECT * FROM shopping_forms WHERE buyer=? ORDER BY created DESC,rowid DESC LIMIT 1",
                    (buyer,),
                ).fetchone()
                return self._public(row) if row else None

        return await asyncio.to_thread(read)

    async def latest(self, buyer, session):
        def read():
            with self._db() as db:
                row = db.execute(
                    "SELECT * FROM shopping_forms WHERE buyer=? AND session=? ORDER BY created DESC,rowid DESC LIMIT 1",
                    (buyer, session),
                ).fetchone()
                return self._public(row) if row else None

        return await asyncio.to_thread(read)

    async def submit(self, buyer, session, form_id, revision, request_id, values):
        if (
            not isinstance(request_id, str)
            or not 1 <= len(request_id) <= 100
            or type(revision) is not int
        ):
            raise ValueError("提交需要请求 ID 和整数版本")

        def write():
            with self._db() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT * FROM shopping_forms WHERE id=? AND buyer=? AND session=?",
                    (form_id, buyer, session),
                ).fetchone()
                form = self._public(row)
                clean = validate_answers(values, form["questions"], complete=True)
                digest = hashlib.sha256(
                    json.dumps(clean, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()
                if row["submission"]:
                    if (
                        row["request_id"] == request_id
                        and row["request_hash"] == digest
                    ):
                        return json.loads(row["submission"])
                    raise FormConflict("该表单已提交，请恢复已保存的需求或新建表单")
                if row["revision"] != revision:
                    raise FormConflict("表单版本已变化，请刷新")
                # 使用持久化的问题定义解释答案，不接受客户端改写题目、选项或单位。
                answers = []
                for question in form["questions"]:
                    key = question["id"]
                    if key not in clean:
                        continue
                    value = clean[key]
                    labels = {o["value"]: o["label"] for o in question["options"]}
                    display = ([labels[v] for v in value] if question["type"] == "multi_select"
                               else labels[value] if question["type"] == "single_select" else value)
                    answers.append({"id": key, "question": question["label"], "value": value,
                                    "display_value": display, "unit": question["unit"]})
                payload = {
                    "form_id": form_id, "title": form["title"],
                    "agent_context": form["context"], "answers": answers,
                    "unanswered": [{"id": q["id"], "question": q["label"]}
                                   for q in form["questions"] if q["id"] not in clean],
                }
                query = ("买家已提交本轮澄清答案。agent_context 仅为 Agent 整理的背景；"
                         "未回答的问题仍未知，以下答案不是交易/记忆审批，也不是已核实的商品或政策事实。\n"
                         + json.dumps(payload, ensure_ascii=False))
                result = {
                    "status": "submitted",
                    "query": query,
                    "revision": revision + 1,
                    "run_id": "form-run-" + uuid.uuid4().hex,
                    "values": clean,
                    "display_text": "\n".join(f"{answer['question']}：{answer['display_value']}" for answer in answers),
                }
                db.execute(
                    "UPDATE shopping_forms SET revision=?,submission=?,request_id=?,request_hash=? WHERE id=?",
                    (
                        revision + 1,
                        json.dumps(result, ensure_ascii=False),
                        request_id,
                        digest,
                        form_id,
                    ),
                )
                return result

        return await asyncio.to_thread(write)
