"""上下文冷证据：完整工具结果保留在 SQLite，模型只持有投影和可校验引用。"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid
from app.application.runtime.request_scope import ShoppingContext


class ContextEvidenceStore:
    def __init__(self, path: Path):
        self.path = path

    async def delete_session(self, buyer: str, session: str) -> None:
        def remove():
            with self._connect() as db:
                db.execute("DELETE FROM context_evidence WHERE buyer=? AND session=?",(buyer,session))
        await asyncio.to_thread(remove)

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=15)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=15000")
            with db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("CREATE TABLE IF NOT EXISTS context_evidence (ref TEXT PRIMARY KEY, buyer TEXT NOT NULL, session TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL, created REAL NOT NULL, fence INTEGER NOT NULL DEFAULT 0)")
                if "fence" not in {row[1] for row in db.execute("PRAGMA table_info(context_evidence)")}:
                    db.execute("ALTER TABLE context_evidence ADD COLUMN fence INTEGER NOT NULL DEFAULT 0")
                db.execute("CREATE INDEX IF NOT EXISTS context_evidence_scope ON context_evidence(buyer,session,kind,created)")
                yield db
        finally:
            db.close()

    async def save(self, buyer: str, session: str, kind: str, payload: dict) -> str:
        if not buyer or not session:
            raise ValueError("证据需要买家与会话作用域")
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode()).hexdigest()
        ref = "ctx_" + uuid.uuid4().hex
        ctx = ShoppingContext.current()
        fence = getattr(ctx, "session_fence", 0) if ctx and ctx.buyer_id == buyer and ctx.shopping_session_id == session else 0
        def write():
            with self._connect() as db:
                db.execute("INSERT INTO context_evidence(ref,buyer,session,kind,payload,sha256,created,fence) VALUES(?,?,?,?,?,?,?,?)", (ref, buyer, session, kind, raw, digest, time.time(), fence))
            return ref
        return await asyncio.to_thread(write)

    async def get(self, buyer: str, session: str, ref: str) -> dict | None:
        def read():
            with self._connect() as db:
                row = db.execute("SELECT * FROM context_evidence WHERE ref=? AND buyer=? AND session=?", (ref, buyer, session)).fetchone()
            return self._decode(row) if row else None
        return await asyncio.to_thread(read)

    async def search(self, buyer: str, session: str, *, kind: str = "", query: str = "", limit: int = 5) -> list[dict]:
        def read():
            with self._connect() as db:
                rows = db.execute("SELECT * FROM context_evidence WHERE buyer=? AND session=? AND kind!='rejected_summary' AND (?='' OR kind=?) AND instr(payload,?)>0 ORDER BY fence DESC,created DESC,ref DESC LIMIT ?", (buyer, session, kind, kind, query, min(10, max(1, limit)))).fetchall()
            return [self._decode(row) for row in rows]
        return await asyncio.to_thread(read)

    async def batch(self, buyer: str, session: str, number: int) -> dict | None:
        """展示批次从1开始，按可信 fence/时间排列，跨买家不可查。"""
        if number < 1:
            raise ValueError("批次序号从1开始")
        def read():
            with self._connect() as db:
                row = db.execute("SELECT * FROM context_evidence WHERE buyer=? AND session=? AND kind='display_batch' ORDER BY fence,created,ref LIMIT 1 OFFSET ?", (buyer, session, number-1)).fetchone()
            return self._decode(row) if row else None
        return await asyncio.to_thread(read)

    async def find_product(self, buyer: str, session: str, *, product_id: str = '', sku_id: str = '') -> dict | None:
        """候选来源仅取真实检索及业务核验后的推荐/比较；展示记录不授予资格。"""
        return await self._find_product(buyer, session, product_id, sku_id,
            ('products','recommendation','comparison'), qualified=True)

    async def find_observed_product(self, buyer: str, session: str, *, product_id: str = '', sku_id: str = '') -> dict | None:
        """只读观察包含权威读取及实际展示；不赋予候选、选购及交易资格。"""
        return await self._find_product(buyer, session, product_id, sku_id,
            ('products','product_view','recommendation','comparison','display_batch'))

    async def _find_product(self, buyer, session, product_id, sku_id, kinds, *, qualified=False):
        if not product_id and not sku_id:
            return None
        def read():
            with self._connect() as db:
                placeholders = ','.join('?' for _ in kinds)
                row = db.execute(f"""
                    SELECT e.* FROM context_evidence e
                    WHERE buyer=? AND session=? AND kind IN ({placeholders})
                    AND EXISTS (SELECT 1 FROM json_each(e.payload, '$.hits') h
                        WHERE (?=0 OR COALESCE(json_array_length(h.value,'$.constraint_issues'),0)=0)
                        AND (?='' OR json_extract(h.value,'$.product_id')=?)
                        AND (?='' OR json_extract(h.value,'$.sku_id')=?
                            OR json_extract(h.value,'$.default_sku_id')=?
                            OR EXISTS (SELECT 1 FROM json_each(h.value,'$.skus') s
                                WHERE json_extract(s.value,'$.sku_id')=?)))
                    ORDER BY fence DESC,created DESC,ref DESC LIMIT 1
                """, (buyer, session, *kinds, int(qualified), product_id, product_id, sku_id, sku_id, sku_id, sku_id)).fetchone()
            return self._decode(row) if row else None
        return await asyncio.to_thread(read)

    @staticmethod
    def _decode(row):
        if hashlib.sha256(row["payload"].encode()).hexdigest() != row["sha256"]:
            raise ValueError("上下文证据校验失败")
        return {"result_ref": row["ref"], "kind": row["kind"], "sha256": row["sha256"], "created_at": row["created"], "session_fence": row["fence"], "data": json.loads(row["payload"])}
