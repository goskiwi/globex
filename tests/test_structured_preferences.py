"""结构化分类与偏好：不猜原话、不扩大材质范围、不丢作用域和原始依据。"""
from tests.shopping_state_helpers import run_search
import json
import sqlite3
from dataclasses import asdict
from types import SimpleNamespace
import pytest
from langchain_core.messages import AIMessage
from app.domain.buyer.preference import BuyerPreference, MaterialExclusion
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.application.agents.shopping_state import ShoppingWork, Filters, compile_search
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.tools.product_search_tool import build_product_search_tool
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.infrastructure.semantic_memory import PreferenceDistiller, SemanticPreferenceStore, MemoryUnavailable
from scripts.migrate_preference_schema import migrate, import_records
from tests.test_semantic_memory import Embed


def preference(statement="不要合成聚合物", category=None):
    return BuyerPreference("b", "dislike", statement,
        constraint=MaterialExclusion(("合成聚合物",),category), evidence=statement)


@pytest.mark.asyncio
async def test_category_is_explicit_not_guessed_from_query():
    class Catalog:
        async def execute(self,spec):
            self.spec=spec
            return {"hits":[],"total_candidates":0,"recall_strategy":"fixture","rerank_applied":False}
    catalog=Catalog()
    tool=build_product_search_tool(catalog,TradeEventBus())
    await run_search(tool, normalized_query="不要耳机，找收纳盒")
    assert catalog.spec.category is None
    await run_search(tool, normalized_query="装耳机的收纳盒", category="家居生活")
    assert catalog.spec.category == "家居生活"
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="category"):
        await run_search(tool, normalized_query="耳机",category="耳机")


def test_scope_and_exceptions_are_data_not_text_matching():
    p=preference(category="厨房餐饮")
    work=ShoppingWork(filters=Filters())
    compiled=compile_search(work,[p])
    assert compiled["parameters"]["excluded_material_tags"] == []
    assert compiled["parameters"]["excluded_materials_by_category"] == {"厨房餐饮":["合成聚合物"]}
    work.ignored_preferences=[p.statement]
    assert compile_search(work,[p])["parameters"]["excluded_materials_by_category"] == {}
    assert p.constraint.category == "厨房餐饮"
    for words in ("不要尼龙","不想买尼龙材质的东西"):
        result=compile_search(ShoppingWork(filters=Filters()),[BuyerPreference("b","dislike",words)])
        assert result["parameters"]["excluded_material_tags"] == []
        assert result["unverified_requirements"] == [words]


@pytest.mark.asyncio
async def test_scoped_material_filter_applies_only_to_matching_products():
    repo=InMemoryProductRepository()
    catalog=CatalogSearchUseCase(repo)
    product=await repo.find_by_id("P1003")
    spec=ProductSearchSpec(product_id="P1003",excluded_materials_by_category={"厨房餐饮":["合成聚合物"]})
    assert catalog.eligible_skus(product,spec)
    spec=ProductSearchSpec(product_id="P1003",excluded_materials_by_category={"旅行装备":["合成聚合物"]})
    assert catalog.sku_constraint_issues(product,product.skus[0],spec) == ["material_excluded"]


@pytest.mark.asyncio
async def test_distiller_persists_constraint_evidence_and_rejects_broadening(tmp_path):
    fact={"kind":"dislike","statement":"厨房餐饮不要合成聚合物","evidence":"厨房餐饮不要合成聚合物",
        "constraint":{"material_tags":["合成聚合物"],"category":"厨房餐饮"},"durable":True,"confidence":.99}
    async def invoke(messages):return AIMessage(content=json.dumps({"facts":[fact]},ensure_ascii=False))
    store=SemanticPreferenceStore(tmp_path/"m.db",PreferenceDistiller(SimpleNamespace(ainvoke=invoke)),Embed(),"test")
    extracted=await store.distiller.extract(BuyerPreference("b","dislike",fact["statement"]))
    assert extracted[0]["constraint"] == {"material_tags":["合成聚合物"],"category":"厨房餐饮"}
    await store.append(BuyerPreference("b","dislike",fact["statement"]))
    saved=(await store.list_by_buyer("b"))[0]
    assert saved.constraint.category == "厨房餐饮" and saved.evidence == fact["evidence"]
    reopened=SemanticPreferenceStore(store.path,None,Embed(),"test")
    assert (await reopened.list_by_buyer("b"))[0] == saved  # 读取不重新提炼。
    fact.update(statement="不要尼龙",evidence="不要尼龙")
    with pytest.raises(MemoryUnavailable):await store.append(BuyerPreference("b","dislike","不要尼龙"))
    assert len(await store.list_by_buyer("b")) == 1


@pytest.mark.asyncio
async def test_offline_migration_preserves_text_and_does_not_invent_conditions(tmp_path):
    path=tmp_path/"old.db"
    with sqlite3.connect(path) as db:
        db.execute('''CREATE TABLE memory_facts(id TEXT PRIMARY KEY,buyer_id TEXT,kind TEXT,statement TEXT,vector TEXT,
                    model_id TEXT,source_hash TEXT,created_at TEXT,version INTEGER,source_kind TEXT,source_ref TEXT)''')
        db.execute("INSERT INTO memory_facts VALUES ('m','b','dislike','不要尼龙','[1,0]','test','source','2026-01-01',3,'user','form')")
    with pytest.raises(MemoryUnavailable):SemanticPreferenceStore(path,None,Embed(),"test")
    migrate(path)
    p=(await SemanticPreferenceStore(path,None,Embed(),"test").list_by_buyer("b"))[0]
    assert p.statement == "不要尼龙" and p.version == 3 and p.constraint is None and p.evidence == ""
    assert compile_search(ShoppingWork(filters=Filters()),[p])["unverified_requirements"] == ["不要尼龙"]
    target=tmp_path/"imported.db"
    rows=[{"buyer_id":"b","kind":"dislike","statement":"不要尼龙","created_at":"2026-01-01"}]
    assert import_records(target,rows) == 1 and import_records(target,rows) == 0
    assert (await SemanticPreferenceStore(target,None,Embed(),"test").list_by_buyer("b"))[0].constraint is None


@pytest.mark.asyncio
async def test_all_stores_preserve_constraint_scope_and_evidence(tmp_path):
    from app.infrastructure.persistence.json_file_stores import JsonFilePreferenceStore
    from app.infrastructure.persistence.sql.repositories import SqlPreferenceStore, create_engine, bootstrap_schema
    engine=create_engine(f"sqlite+aiosqlite:///{tmp_path/'plain.db'}")
    await bootstrap_schema(engine)
    try:
        for store in (JsonFilePreferenceStore(tmp_path),SqlPreferenceStore(engine)):
            p=preference("厨房餐饮不要合成聚合物",category="厨房餐饮")
            await store.append(p)
            actual=(await store.list_by_buyer("b"))[0]
            assert actual.constraint == p.constraint and actual.evidence == p.evidence
            changed=preference("旅行装备不要合成聚合物",category="旅行装备")
            assert await store.replace("b",p.statement,changed)
            assert (await store.list_by_buyer("b"))[0].constraint.category == "旅行装备"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_unknown_material_does_not_silently_become_a_noop_filter():
    with pytest.raises(ValueError,match="材质条件"):
        ProductSearchSpec(product_id="P1003",excluded_material_tags=["尼龙"])
