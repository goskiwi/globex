"""显式升级实际旧表，保留记录，不用默认报价掩盖历史费用缺失。"""
import sqlite3

import pytest
from sqlalchemy import create_engine

from scripts.migrate_pricing_schema import migrate as migrate_pricing
from scripts.migrate_preference_schema import migrate as migrate_preferences
from app.infrastructure.persistence.sql.tables import Base
import app.infrastructure.persistence.sql.trade_tables
import app.infrastructure.persistence.sql.session_store


def old_database(path):
    engine = create_engine('sqlite:///'+str(path))
    Base.metadata.create_all(engine)
    engine.dispose()
    with sqlite3.connect(path) as db:
        db.execute('ALTER TABLE orders DROP COLUMN pricing_json')
        db.execute('ALTER TABLE buyer_preferences DROP COLUMN constraint_json')
        db.execute('ALTER TABLE buyer_preferences DROP COLUMN evidence')
        db.execute("INSERT INTO conversation_sessions (session_id,buyer_id,locale,currency) VALUES ('test-session','test-buyer','zh-CN','CNY')")
        db.execute("INSERT INTO trade_sku_inventory VALUES ('P1003-S1','P1003','测试商品',80,12900,'CNY')")


def test_empty_trade_schema_and_preference_schema_upgrade_preserve_other_tables(tmp_path):
    path = tmp_path/'old.db'
    old_database(path)
    assert '已升级' in migrate_pricing(path)
    migrate_preferences(path)
    with sqlite3.connect(path) as db:
        assert 'pricing_json' in {r[1] for r in db.execute('PRAGMA table_info(orders)')}
        preferences = {r[1]:r for r in db.execute('PRAGMA table_info(buyer_preferences)')}
        assert preferences['constraint_json'][2] == 'JSON' and preferences['constraint_json'][3] == 0
        assert preferences['evidence'][3] == 1
        assert db.execute('SELECT COUNT(*) FROM orders').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM conversation_sessions').fetchone()[0] == 1
        assert db.execute('SELECT stock,unit_price_minor FROM trade_sku_inventory').fetchone() == (80,12900)
        assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        assert db.execute('PRAGMA foreign_key_check').fetchall() == []
    assert migrate_pricing(path) == '无需迁移'
    migrate_preferences(path)


def test_orphan_order_item_is_not_treated_as_empty_history(tmp_path):
    path = tmp_path/'old.db'
    old_database(path)
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO order_items (order_id,product_id,sku_id,title,unit_price_minor,currency,quantity) VALUES ('orphan','P1003','P1003-S1','测试',12900,'CNY',1)")
    with pytest.raises(ValueError, match='未修改数据库'):
        migrate_pricing(path)
    with sqlite3.connect(path) as db:
        assert 'pricing_json' not in {r[1] for r in db.execute('PRAGMA table_info(orders)')}
        assert db.execute('SELECT COUNT(*) FROM order_items').fetchone()[0] == 1


def test_preference_migration_keeps_existing_text_without_inventing_constraints(tmp_path):
    path = tmp_path/'old.db'
    old_database(path)
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO buyer_preferences (buyer_id,kind,statement,created_at) VALUES ('test-buyer','like','喜欢轻便','2026-10-01')")
    migrate_preferences(path)
    with sqlite3.connect(path) as db:
        row = db.execute('SELECT statement,constraint_json,evidence FROM buyer_preferences').fetchone()
    assert row == ('喜欢轻便',None,'')
