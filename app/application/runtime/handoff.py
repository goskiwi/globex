"""子任务交付生命周期：候选提交 → 整轮校验 → 结果或有限修正。

复用 create_agent 的工具节点及 before/after_model 条件路由，不维护第二套执行循环。
"""
from dataclasses import dataclass
from copy import deepcopy
import json
from typing_extensions import NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from app.application.agents.handoff import AgentSubmission, SubagentResult
from app.application.runtime.errors import ExecutionStopped, raise_if_tool_stopped
from app.application.runtime.results import message_data


def issue(code, field, message):
    return {"code": code, "field": field, "message": message}


class TaskEvidence:
    """当前子调用实际产生的证据；输入中提到对象不等于获得了推荐依据。"""
    def __init__(self, session_id):
        self.session_id = session_id
        self.refs = set()
        self.facts = {}
        self.quotes = {}
        self.identifiers = set()
        self.trade = []
        self.successful_tools = set()
        self.historical_refs = set()
        self._lookups = set()
        self.delegated = {}

    @property
    def products(self):
        # 校验和交付读取同一份事实索引，不维护一份会与快照分离的 SKU 集合。
        result = {}
        for product_id, sku_id in self.facts:
            result.setdefault(product_id, set())
            if sku_id is not None:
                result[product_id].add(sku_id)
        return result

    def _products(self, cards, result_ref=None, historical=False, observed_at=None):
        product_fields = {"product_id", "title", "brand", "category", "origin_country",
            "description", "highlights", "material_tags", "weight_kg", "dimensions_cm", "package_dimensions_cm",
            "rating_summary", "rating_is_live", "ships_to", "updated_at",
            "source_platform", "canonical_product_id", "data_provenance"}
        for card in cards:
            product_id = card["product_id"]
            common = {key:deepcopy(value) for key,value in card.items() if key in product_fields}
            # 商品级资料不承载任何默认 SKU 的金额或库存。
            for sku in [None, *card.get("skus", [])]:
                key = (product_id, sku["sku_id"] if sku else None)
                old = self.facts.get(key)
                if old and not old["historical"] and historical:
                    continue
                facts = {"result_ref":result_ref, "historical":historical,
                         "observed_at":observed_at, "product":common,
                         "sku":deepcopy(sku)}
                self.facts[key] = facts
                self.identifiers.add(product_id)
                if sku:
                    self.identifiers.add(sku["sku_id"])
            self._quotes(card.get("landed_price") or {}, result_ref, historical, observed_at)

    def _quotes(self, quote, result_ref, historical, observed_at):
        for line in quote.get("items", []):
            key = (line["product_id"], line["sku_id"], line["quantity"], quote["ship_to"], quote["currency"])
            old = self.quotes.get(key)
            if old and not old["historical"] and historical:
                continue
            self.quotes[key] = {"result_ref": result_ref, "historical": historical,
                "observed_at": observed_at, "ship_to": quote["ship_to"], "currency": quote["currency"],
                "line": deepcopy(line)}

    def capture_lookup(self, messages):
        for message in messages:
            if (not isinstance(message, ToolMessage) or message.name != "conversation_fact_lookup"
                    or message.status == "error"):
                continue
            data = message_data(message)
            if not isinstance(data, dict) or data.get("source") != "session_evidence":
                continue
            for record in data.get("records", []):
                key = (message.tool_call_id, record["result_ref"])
                if key in self._lookups: continue
                self._lookups.add(key)
                self.refs.add(record["result_ref"])
                self.historical_refs.add(record["result_ref"])
                if record.get("kind") in {"products", "display_batch", "recommendation", "comparison"}:
                    self._products(record.get("data", {}).get("hits", []), record["result_ref"], True,
                                   record.get("data", {}).get("observed_at"))
                if record.get("kind") == "quote":
                    self._quotes(record.get("data", {}), record["result_ref"], True,
                                 record.get("data", {}).get("observed_at"))
            self.successful_tools.add("conversation_fact_lookup")

    def capture(self, event):
        if event.shopping_session_id != self.session_id or event.type != "tool.result":
            return
        payload = event.payload
        if not isinstance(payload, dict):
            return
        name = payload.get("tool")
        if name == "task_dispatch":
            self.delegated[payload.get("tool_call_id") or str(len(self.delegated))] = {
                "agent": payload.get("agent"), "status": payload.get("status"),
                "stop_reason": payload.get("stop_reason"),
                "candidates": [c["product_id"] for c in payload.get("candidates", [])]}
            return
        if payload.get("error"):
            return
        if name == "conversation_fact_lookup":
            self.capture_lookup([ToolMessage(content="", name=name, tool_call_id=payload.get("tool_call_id","lookup"),
                id=payload.get("tool_call_id"), artifact={"data":payload})])
            return
        if name not in {"product_search_tool", "get_product_details", "category_insight_tool", "web_search_tool",
                        "query_order_tool", "create_order_tool", "cancel_order_tool", "quote_products"}:
            return
        self.successful_tools.add(name)
        if payload.get("result_ref"):
            self.refs.add(payload["result_ref"])
        self._products(payload.get("hits", []), payload.get("result_ref"), False, payload.get("observed_at"))
        if name == "quote_products":
            self._quotes(payload.get("quote", {}), payload.get("result_ref"), False, payload.get("observed_at"))
        confirmation = payload.get("confirmation") or {}
        order = payload if name == "query_order_tool" else payload.get("order") or {}
        for record in [confirmation.get("payload") or {}, order]:
            for line in record.get("items", record.get("lines", [])):
                self.identifiers.update(str(line[k]) for k in ("product_id", "sku_id") if line.get(k))
        if confirmation:
            self.trade.append({"confirmation_id": confirmation["confirmation_id"],
                "action": confirmation["action"], "status": confirmation["status"],
                "expired": confirmation.get("expired", False)})
        if order:
            self.trade.append({"order_id": order["order_id"], "status": order["status"]})

    def stopped_result(self, reason):
        """停止时交回证据与缺口；检索命中不能自动升级为已筛选候选。"""
        return self.delivered(AgentSubmission(
            status="partial" if self.successful_tools else "failed",
            summary=str(ExecutionStopped(reason)) + "；任务尚未完整完成。",
            issues=[reason]))

    def stopped_text(self, reason):
        result = self.stopped_result(reason)
        lines = [result.summary]
        if self.products:
            lines.append(f"已读取 {len(self.products)} 件商品，完整资料保留在下列证据中；尚未完成筛选，不能当作推荐。")
        elif "product_search_tool" in self.successful_tools:
            lines.append("已完成的检索没有返回候选，不代表商品不存在。")
        elif not self.successful_tools:
            lines.append("本轮尚无可核验的业务成果。")
        other_reads = self.successful_tools & {"category_insight_tool", "web_search_tool", "conversation_fact_lookup"}
        if other_reads:
            lines.append("已执行部分资料查询，尚未形成完整结论。")
        if self.refs:
            lines.append("已保留证据引用：" + "、".join(sorted(self.refs)) + "。")
        if self.historical_refs:
            lines.append("其中历史证据不能替代当前价格、库存或订单状态。")
        # 同一实体按最后一次已核验结果展示，不把早先待确认状态覆盖到后续状态。
        trades = {r.get("confirmation_id") or r.get("order_id"): r for r in self.trade}
        for item in trades.values():
            if item.get("confirmation_id"):
                pending = item["status"] == "pending" and not item.get("expired")
                lines.append(f"确认单 {item['confirmation_id']}：" +
                             ("待用户确认，尚未执行交易。" if pending else f"状态 {item['status']}，请核对页面记录。"))
            else:
                lines.append(f"订单 {item['order_id']}：已核验状态 {item['status']}。")
        for task in self.delegated.values():
            lines.append(f"子任务 {task['agent']}：{task['status']}" +
                         (f"（{task['stop_reason']}）" if task["stop_reason"] else "") + "。")
            if task["candidates"]:
                lines.append("该子任务明确提交的候选：" + "、".join(task["candidates"]) + "；尚未由 Main 交付最终推荐。")
        lines.append("未完成部分需要继续核对；涉及写入但未取得结果的操作，请先查询状态，不要直接重放。")
        return "\n".join(lines)

    def delivered(self, submission):
        """只关联本子任务实际读取的商品与引用；模型不能填写或覆盖事实。"""
        payload = submission.model_dump()
        for candidate in payload["candidates"]:
            key = (candidate["product_id"], candidate.get("sku_id"))
            candidate["facts"] = deepcopy(self.facts[key])
            candidate["facts"]["quotes"] = [deepcopy(value) for quote_key,value in self.quotes.items()
                if key[1] is not None and quote_key[:2] == key]
        return SubagentResult(**payload, evidence_refs=sorted(self.refs),
            observed_product_count=len(self.products),
            observed_sku_count=sum(len(skus) for skus in self.products.values()))

    def validate(self, result):
        errors = []
        # 只校验声明为推荐候选的对象，不扫描 summary/questions 等自由文本。
        for index, candidate in enumerate(result.candidates):
            if candidate.product_id not in self.products:
                errors.append(issue("candidate_unverified", f"candidates.{index}.product_id",
                                    "该候选没有实际读取证据，请移除，或将缺口交回主 Agent。"))
            elif candidate.sku_id is not None and candidate.sku_id not in self.products[candidate.product_id]:
                errors.append(issue("sku_mismatch", f"candidates.{index}.sku_id",
                                    "该 SKU 不属于候选商品的已读证据，请按工具结果修正。"))
        if result.status == "completed" and not self.successful_tools:
            errors.append(issue("completion_without_evidence", "status",
                                "没有成功业务工具结果；如需用户补充或无法完成，请如实修改状态。"))
        return errors


@dataclass
class HandoffContext:
    evidence: TaskEvidence


class HandoffState(AgentState):
    handoff_task: NotRequired[dict]
    handoff_submission: NotRequired[list[dict]]
    handoff_result: NotRequired[dict | None]
    handoff_feedback: NotRequired[list[dict]]
    handoff_attempts: NotRequired[int]


class HandoffMiddleware(AgentMiddleware[HandoffState, HandoffContext]):
    state_schema = HandoffState
    max_corrections = 2

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        context = request.runtime.context
        if isinstance(context, HandoffContext) and isinstance(result, ToolMessage):
            context.evidence.capture_lookup([result])
        return result

    def _reject(self, state, errors, replacements=None):
        attempts = state.get("handoff_attempts", 0) + 1
        update = {"handoff_submission": [], "handoff_feedback": errors,
                  "handoff_attempts": attempts, "messages": replacements or []}
        if attempts > self.max_corrections:
            update.update(handoff_result=SubagentResult(status="failed",
                summary="子任务提交未通过校验，已停止修正；已完成业务结果仍保留。",
                issues=[f"{e['code']} ({e['field']}): {e['message']}" for e in errors]).model_dump(),
                jump_to="end")
        else:
            update["messages"].append(HumanMessage(name="handoff_feedback", content=json.dumps({
                "errors": errors, "instruction": "仅依据已有结果修正 SubagentResult；不要重新执行业务工具。",
                "corrections_remaining": self.max_corrections - attempts + 1}, ensure_ascii=False)))
            update["jump_to"] = "model"
        return update

    @hook_config(can_jump_to=["model", "end"])
    async def aafter_model(self, state, runtime):
        last = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        submissions = [c for c in last.tool_calls if c["name"] == "SubagentResult"] if last else []
        rejected = []
        if state.get("handoff_attempts") and last:
            # 修正交付时原业务工具已执行过，不允许重放确认准备等操作。
            rejected = [ToolMessage(name=c["name"], tool_call_id=c["id"], status="error",
                content="提交修正阶段未执行业务工具；请使用已有结果修正提交。")
                for c in last.tool_calls if c["name"] != "SubagentResult"]
        if submissions:
            return {"handoff_submission": submissions, "messages": rejected}
        if not last or not last.tool_calls or rejected:
            return self._reject(state, [issue("submission_missing", "submission",
                "请调用 SubagentResult 交付，普通文本不作为最终成果。")], rejected)
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        raise_if_tool_stopped(state["messages"])
        submissions = state.get("handoff_submission") or []
        if not submissions:
            return None
        context = runtime.context
        if not isinstance(context, HandoffContext):
            raise RuntimeError("子任务缺少服务端证据上下文")
        evidence = context.evidence
        evidence.capture_lookup(state["messages"])
        if len(submissions) != 1:
            errors = [issue("multiple_submissions", "submission", "一次只提交一份最终结果。")]
            result = None
        else:
            try:
                receipt = next((m for m in reversed(state["messages"])
                    if isinstance(m, ToolMessage) and m.tool_call_id == submissions[0]["id"]), None)
                if receipt is None or receipt.status == "error":
                    result = None
                    errors = [issue("submission_delivery_failed", "submission", receipt.text if receipt is not None else "提交工具未成功接收，请修正提交。")]
                else:
                    # 按 call_id 读取经现有工具中间件处理后的候选，避免绕过内容过滤。
                    payload = message_data(receipt)
                    result = AgentSubmission.model_validate(payload)
                    errors = evidence.validate(result)
            except ValidationError as error:
                result = None
                errors = [issue("invalid_field", ".".join(map(str, e["loc"])) or "submission", e["msg"])
                          for e in error.errors(include_input=False, include_url=False, include_context=False)]
            except (ValueError, TypeError):
                result = None
                errors = [issue("invalid_submission_format", "submission", "提交工具未返回合法 JSON 对象，请重新提交。")]
        call_ids = {s["id"] for s in submissions}
        # 替换提交工具的接收回执，不依赖它在整批消息中的位置。
        replacements = [m.model_copy(update={"status": "error" if errors else "success",
            "content": json.dumps({"accepted": not errors, "errors": errors}, ensure_ascii=False)})
            for m in state["messages"] if isinstance(m, ToolMessage) and m.tool_call_id in call_ids]
        if errors:
            update = self._reject(state, errors, replacements)
            if update["jump_to"] == "model":
                # 此节点本身就在模型前，沿默认边进入模型，不能跳回自身。
                update.pop("jump_to")
            return update
        reason = state.get("execution_stop")
        if reason and result.status == "completed":
            result = result.model_copy(update={"status": "partial", "issues": [*result.issues, reason]})
        return {"handoff_result": evidence.delivered(result).model_dump(), "handoff_submission": [],
                "handoff_feedback": [], "messages": replacements, "jump_to": "end"}

    async def awrap_model_call(self, request, handler):
        if request.state.get("handoff_attempts"):
            return await handler(request.override(
                tools=[t for t in request.tools if t.name == "SubagentResult"], tool_choice="auto"))
        return await handler(request)
