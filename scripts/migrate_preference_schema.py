"""离线迁移偏好结构；只添加未知条件标记，不调用模型，不补造排除条件。"""
import argparse
import sys
import sqlite3
import json
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def migrate(path: Path):
    with sqlite3.connect(f"file:{path.resolve()}?mode=rw", uri=True) as db:
        db.execute("BEGIN IMMEDIATE")
        tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        changed=[]
        for table in ("memory_facts", "buyer_preferences"):
            if table not in tables:
                continue
            columns={row[1] for row in db.execute(f"PRAGMA table_info({table})")}
            if "constraint_json" not in columns:
                definition = "JSON" if table == "buyer_preferences" else "TEXT NOT NULL DEFAULT 'null'"
                db.execute(f"ALTER TABLE {table} ADD COLUMN constraint_json {definition}")
            if "evidence" not in columns:
                db.execute(f"ALTER TABLE {table} ADD COLUMN evidence TEXT NOT NULL DEFAULT ''")
            changed.append({"table":table,"records":db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]})
        if not changed:
            raise ValueError("指定数据库没有偏好表，未做迁移")
        return changed


def import_records(target: Path, records):
    """显式导入旧文本，保留原文与时间；没有依据的执行条件保持 null。"""
    from app.infrastructure.semantic_memory import SemanticPreferenceStore
    from app.domain.buyer.preference import BuyerPreference
    store=SemanticPreferenceStore(target,None,None,"unindexed")
    inserted=0
    with store._db() as db:
        db.execute("BEGIN IMMEDIATE")
        for row in records:
            if row.get("constraint") is not None or row.get("constraint_json") not in (None,"null"):
                raise ValueError("来源已含执行条件，不能作为旧文本重新导入")
            p=BuyerPreference(row["buyer_id"],row["kind"],row["statement"],created_at=row.get("created_at", ""))
            identifier=uuid.uuid4().hex
            count=db.execute('''INSERT OR IGNORE INTO memory_facts
                (id,buyer_id,kind,statement,vector,model_id,source_hash,created_at,version,source_kind,source_ref,constraint_json,evidence)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''', (identifier,p.buyer_id,p.kind,p.statement,"[]","unindexed",
                hashlib.sha256((p.kind+'\0'+p.statement).encode()).hexdigest(),p.created_at,1,"legacy","offline_import","null","")).rowcount
            if count:
                store._audit(db,p.buyer_id,identifier,"create",1,"legacy","offline_import",p.statement)
            inserted+=count
    return inserted


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database",type=Path)
    source=parser.add_mutually_exclusive_group()
    source.add_argument("--import-sql",type=Path,help="只读导入旧 buyer_preferences 表")
    source.add_argument("--import-json",type=Path,help="只读导入一个旧偏好 JSON 文件")
    args=parser.parse_args()
    if args.database.exists(): print(migrate(args.database))
    if args.import_sql:
        with sqlite3.connect(f"file:{args.import_sql.resolve()}?mode=ro",uri=True) as db:
            db.row_factory=sqlite3.Row
            rows=[dict(row) for row in db.execute("SELECT * FROM buyer_preferences")]
        print({"imported":import_records(args.database,rows)})
    elif args.import_json:
        print({"imported":import_records(args.database,json.loads(args.import_json.read_text()))})
    elif not args.database.exists():
        parser.error("结构迁移必须指定已存在的数据库")
