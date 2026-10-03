"""实际导出 Schema、工具执行和委派条件必须使用同一份输入定义。"""
import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from langchain_core.utils.function_calling import convert_to_openai_tool
from app.application.runtime.tools import as_langchain_tool
from app.application.agents.handoff import DelegatedTask, build_submission_tool
from app.application.agents.shopping_state import ShoppingWork, Filters, compile_search
from app.application.tools.product_search_tool import build_product_search_tool, SearchQuery
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus


def test_exported_search_schema_matches_runtime_and_rejects_old_fields():
    schema=convert_to_openai_tool(as_langchain_tool(build_product_search_tool(None,TradeEventBus())))['function']['parameters']
    assert set(schema['properties'])=={'normalized_query','category','top_k','product_id','sku_id'}
    assert schema['additionalProperties'] is False and len(schema['anyOf'])==3
    assert schema['properties']['top_k']['minimum']==1
    assert schema['properties']['top_k']['maximum']==50
    for value in ({'normalized_query':'包','price_max_major':300}, {}, {'normalized_query':'包','top_k':51}):
        with pytest.raises(ValidationError): SearchQuery.model_validate(value)


def test_delegation_and_submission_have_no_duplicate_input_channels():
    with pytest.raises(ValidationError):
        DelegatedTask(goal='研究',search_filters={'price_max_major':300})
    with pytest.raises(ValidationError):
        DelegatedTask(goal='研究',hard_constraints=['预算300'],expected_output='价格')
    schema=convert_to_openai_tool(build_submission_tool())['function']['parameters']
    assert 'evidence_refs' not in schema['properties'] and 'notes' not in schema['properties']
    assert 'facts' not in schema['properties']['candidates']['items'].get('properties',{})
    assert schema['additionalProperties'] is False


async def test_effective_conditions_reach_actual_search_without_model_recopy():
    class Catalog:
        async def execute(self,spec):
            self.spec=spec
            return {'hits':[],'total_candidates':0,'recall_strategy':'fixture','rerank_applied':False}
    catalog=Catalog(); bus=TradeEventBus(); search=build_product_search_tool(catalog,bus)
    class Factory:
        def build(self):
            async def invoke(inputs,**kwargs):
                self.sent=json.loads(inputs['messages'][-1].content)
                self.result=(await search(normalized_query='背包')).data
                return {'handoff_result':{'status':'needs_input','summary':'需补充','questions':['用途？']}}
            return SimpleNamespace(ainvoke=invoke)
    factory=Factory()
    work=ShoppingWork(filters=Filters(ship_to='CN',target_currency='CNY',price_max_major=500,
                                     excluded_material_tags=['金属']))
    snapshot=ShoppingContextSnapshot('s','b','zh-CN','CNY',effective_search=compile_search(work))
    token=ShoppingContext.set(snapshot)
    try:
        await build_task_dispatch_tool(factory,factory,bus)('search_agent',
            {'goal':'背包研究','filters':{'price_max_major':200}},
            runtime=SimpleNamespace(state={'shopping_work':work.model_dump()},tool_call_id='task'))
        assert catalog.spec.price_max_major==200 and catalog.spec.ship_to=='CN'
        assert catalog.spec.excluded_material_tags==['金属']
        assert factory.result['query_conditions']['price_max_major']==200
        assert factory.sent['parent_context']['effective_search']['parameters']['price_max_major']==200
        assert 'filters' not in factory.sent['delegated_task']
        assert ShoppingContext.current() is snapshot and work.filters.price_max_major==500
        with pytest.raises(TypeError): await search(normalized_query='背包',price_max_major=1)
    finally:
        ShoppingContext.reset(token)


async def test_search_without_authoritative_conditions_fails_closed():
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    try:
        result=await build_product_search_tool(None,TradeEventBus())(normalized_query='背包')
        assert not result.ok and '有效购物条件' in (result).text
    finally:
        ShoppingContext.reset(token)
