"""离线升级空交易库；历史商品金额单据不可凭当前规则补造运税。"""
import argparse
import sqlite3
from pathlib import Path


def migrate(path: Path):
    with sqlite3.connect(f"file:{path.resolve()}?mode=rw", uri=True) as db:
        db.execute("BEGIN IMMEDIATE")
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {r[1] for r in db.execute("PRAGMA table_info(orders)")}
        if "orders" not in tables or "pricing_json" in columns:
            return "无需迁移"
        populated = [name for name in ("orders", "order_items", "trade_confirmations", "trade_operations")
                     if name in tables and db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]]
        if populated:
            raise ValueError("存在历史交易记录，未修改数据库。旧单据不能补造运税；请先决定归档旧库还是提供经核对的历史金额。")
        db.execute("ALTER TABLE orders ADD COLUMN pricing_json JSON NOT NULL")
        return "空交易库已升级，未删除任何数据"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    print(migrate(parser.parse_args().database))
