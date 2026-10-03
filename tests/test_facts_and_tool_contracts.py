"""商品/SKU/报价来源、真实导出 Schema 与研究交付语义的集中反例。"""
import json
import pytest
from pydantic import ValidationError
from langchain_core.utils.function_calling import convert_to_openai_tool
from app.application.agents.handoff import AgentSubmission, build_submission_tool
from app.application.runtime.handoff import TaskEvidence
from app.application.runtime.tools import as_langchain_tool
from app.application.tools.recommendation_tools import build_quote_tool, QuoteInput
from app.application.tools.order_tools import build_create_order_tool
from app.application.tools.order_inputs import CreateOrderInput, ShippingAddressInput
from app.application.tools.shopping_state_tool import build_shopping_state_tool
from app.application.agents.shopping_state import ShoppingWork, Filters, ShoppingUpdate, apply_update, compile_search
from app.infrastructure.eventbus import TradeEventBus


def line(sku, amount, quantity=1, currency='CNY'):
    return {'product_id':'P1003','sku_id':sku,'quantity':quantity,
            'unit_price_minor':amount,'total_amount_minor':amount*quantity+2500,'currency':currency}


def card():
    return {'product_id':'P1003','title':'包','default_sku_id':'P1003-S1',
        'price_major':100,'currency':'CNY','source_price_major':100,'source_currency':'CNY',
        'skus':[{'sku_id':'P1003-S1','price_major':100,'currency':'CNY','stock':3},
                {'sku_id':'P1003-S2','price_major':200,'currency':'CNY','stock':2}],
        'landed_price':{'items':[line('P1003-S1',10000)],'ship_to':'CN','currency':'CNY'}}


def deliver(evidence, sku='P1003-S2'):
    submission=AgentSubmission(status='completed',summary='研究结果',candidates=[
        {'product_id':'P1003','sku_id':sku,'reason':'用途匹配'}])
    assert not evidence.validate(submission)
    return evidence.delivered(submission).candidates[0].facts


def test_alternative_sku_never_inherits_default_sku_amount():
    evidence=TaskEvidence('s'); evidence.successful_tools.add('product_search_tool')
    evidence._products([card()],'all',observed_at='t1')
    facts=deliver(evidence)
    assert facts['sku']['sku_id']=='P1003-S2' and facts['sku']['price_major']==200
    assert facts['quotes']==[]
    assert not {'price_major','default_sku_id','landed_price','currency'} & facts['product'].keys()
    first=deliver(evidence,'P1003-S1')
    assert first['quotes'][0]['line']['sku_id']=='P1003-S1'
    assert first['quotes'][0]['result_ref']=='all'


def test_narrow_read_keeps_other_sku_and_validation_matches_delivery():
    evidence=TaskEvidence('s'); evidence.successful_tools.add('product_search_tool')
    evidence._products([card()],'all',observed_at='t1')
    narrow=card();narrow['skus']=narrow['skus'][:1]
    evidence._products([narrow],'s1-only',observed_at='t2')
    facts=deliver(evidence)
    assert facts['sku']['sku_id']=='P1003-S2' and facts['result_ref']=='all'
    assert deliver(evidence,'P1003-S1')['result_ref']=='s1-only'


def test_history_does_not_replace_current_sku_or_quote():
    evidence=TaskEvidence('s'); evidence.successful_tools.add('product_search_tool')
    evidence._products([card()],'current')
    old=card();old['skus'][0]['price_major']=1
    old['landed_price']['items'][0]['unit_price_minor']=100
    evidence._products([old],'history',historical=True)
    facts=deliver(evidence,'P1003-S1')
    assert facts['result_ref']=='current' and facts['sku']['price_major']==100
    assert facts['quotes'][0]['result_ref']=='current'


def test_multi_item_quotes_keep_line_identity_and_quantity_currency_destination():
    evidence=TaskEvidence('s'); evidence.successful_tools.add('product_search_tool')
    evidence._products([card()],'products')
    evidence._quotes({'items':[line('P1003-S1',10000),line('P1003-S2',20000,2)],
                      'ship_to':'CN','currency':'CNY'},'bundle',False,'t2')
    evidence._quotes({'items':[line('P1003-S2',3000,currency='USD')],'ship_to':'US','currency':'USD'},'usd',False,'t3')
    quotes=deliver(evidence)['quotes']
    assert len(quotes)==2 and all(q['line']['sku_id']=='P1003-S2' for q in quotes)
    assert {(q['line']['quantity'],q['ship_to'],q['currency']) for q in quotes}=={(2,'CN','CNY'),(1,'US','USD')}
    assert quotes[0]['line']['total_amount_minor']==42500


def test_product_only_result_has_no_sku_price():
    evidence=TaskEvidence('s'); evidence.successful_tools.add('product_search_tool')
    evidence._products([card()],'products')
    facts=deliver(evidence,None)
    assert facts['sku'] is None and facts['quotes']==[]


def test_quote_and_address_schemas_are_nested_and_enforced():
    schema=convert_to_openai_tool(as_langchain_tool(build_quote_tool(None,None,None)))['function']['parameters']
    item=schema['properties']['items']['items']
    assert set(item['required'])=={'product_id','sku_id','quantity'}
    assert item['properties']['quantity']['exclusiveMinimum']==0 and item['additionalProperties'] is False
    for quantity in (0,-1,True,'2'):
        with pytest.raises(ValidationError):
            QuoteInput(items=[{'product_id':'P1003','sku_id':'P1003-S1','quantity':quantity}],ship_to='CN',currency='CNY')
    with pytest.raises(ValidationError): QuoteInput(items=[],ship_to='CN',currency='CNY')
    order=convert_to_openai_tool(as_langchain_tool(build_create_order_tool(None,TradeEventBus())))['function']['parameters']
    address=order['properties']['shipping_address']
    assert set(address['required'])=={'recipient_name','country','city','address_line'}
    assert address['additionalProperties'] is False
    with pytest.raises(ValidationError): CreateOrderInput(sku_ids=['P1003-S1'],shipping_address={})


def test_shopping_tool_example_actually_enforces_budget():
    description=build_shopping_state_tool(None).description
    payload=json.loads(next(line.strip().removesuffix('。') for line in description.splitlines() if line.strip().startswith('{"update"')))
    update=ShoppingUpdate.model_validate(payload['update'])
    state=apply_update(ShoppingWork(filters=Filters(),source_message_id='buyer-example'),update,'buyer-example')
    compiled=compile_search(state)
    assert compiled['parameters']['price_max_major']==300 and compiled['parameters']['landed_budget_major'] is None
    assert compiled['parameters']['target_currency']=='CNY' and not compiled['unverified_requirements']


def test_optional_phone_is_plain_optional_text_without_format_conversion():
    address={'recipient_name':'合成买家','country':'CN','city':'测试市','address_line':'测试路1号'}
    assert ShippingAddressInput(**address).phone==''
    for phone in ('', '+86 123-456', '(123) 456.7890', '+1 555 1234 ext 99', '留空'):
        assert ShippingAddressInput(**address,phone=phone).phone==phone
    assert 'pattern' not in ShippingAddressInput.model_json_schema()['properties']['phone']


def test_unknown_research_attributes_are_not_unmet_user_constraints():
    result=AgentSubmission(status='completed',summary='研究结果',unknowns=['未标注电脑隔层'])
    assert result.status=='completed' and not result.unmet_constraints
    with pytest.raises(ValidationError):
        AgentSubmission(status='completed',summary='研究结果',unmet_constraints=['买家要求防水，未核验'])


async def test_delegation_keeps_prior_conditions_and_current_source_without_substring_gate():
    from tests.test_subagent_preference_inject import RecordingFactory
    from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
    from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
    from langchain_core.messages import HumanMessage
    from types import SimpleNamespace
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    factory=RecordingFactory()
    work=ShoppingWork(filters=Filters(ship_to='CN'),unverified_requirements=['不要防水款'])
    runtime=SimpleNamespace(state={'shopping_work':work.model_dump(),
        'messages':[HumanMessage(content='继续研究，条件不变',name='b')]},tool_call_id='task')
    tool=build_task_dispatch_tool(factory,factory,TradeEventBus())
    try:
        result=await tool('search_agent',{'goal':'研究','requirements':['不要防水款'],
            'research_dimensions':['电脑隔层']},runtime=runtime)
        assert result.ok
        assert factory.seen[0]['parent_context']['effective_search']['unverified_requirements']==['不要防水款']
        assert factory.seen[0]['parent_context']['latest_user_request']=='继续研究，条件不变'
        assert factory.seen[0]['delegated_task']['research_dimensions']==['电脑隔层']
    finally:
        ShoppingContext.reset(token)


@pytest.mark.parametrize('sku', ['', ' ', '\t\n'])
def test_blank_sku_is_rejected_not_converted_to_product_only(sku):
    with pytest.raises(ValidationError):
        AgentSubmission(status='completed',summary='候选',candidates=[
            {'product_id':'P1003','sku_id':sku,'reason':'用途匹配'}])


def test_known_sku_and_product_only_candidates_have_real_facts():
    evidence=TaskEvidence('s'); evidence.successful_tools.add('product_search_tool')
    evidence._products([card()],'products')
    for sku in (None, 'P1003-S1', 'P1003-S2'):
        facts=deliver(evidence,sku)
        assert facts['product']['product_id']=='P1003'
        assert facts['sku'] is None if sku is None else facts['sku']['sku_id']==sku
    schema=convert_to_openai_tool(build_submission_tool())['function']['parameters']
    sku_schema=schema['properties']['candidates']['items']['properties']['sku_id']
    assert any(branch.get('minLength')==1 for branch in sku_schema['anyOf'])


async def test_production_tool_inventory_has_no_opaque_objects(tmp_path, monkeypatch):
    from tests.test_retrieval import _settings
    from tests.test_langgraph_runtime import ScriptedModel
    from scripts.eval.interview_runtime import isolated_runtime
    import app.application.agents.main_agent as main_module
    original=main_module.create_agent
    observed=[]
    def capture(*args, **kwargs):
        observed.extend(kwargs['tools'])
        return original(*args, **kwargs)
    monkeypatch.setattr(main_module,'create_agent',capture)
    async with isolated_runtime(tmp_path/'runtime',_settings(tmp_path),model_factory=lambda *a,**kw:ScriptedModel()) as (_,container):
        container.orchestrator._sessions._main_factory.build()
    def inspect_schema(schema):
        if isinstance(schema, dict):
            if schema.get('type')=='object':
                assert 'properties' in schema
            for value in schema.values(): inspect_schema(value)
        elif isinstance(schema,list):
            for value in schema: inspect_schema(value)
    names=set()
    for tool in observed:
        schema=convert_to_openai_tool(tool)['function']
        names.add(schema['name'])
        assert schema['parameters']['additionalProperties'] is False, schema['name']
        inspect_schema(schema['parameters'])
    assert {'quote_products','create_order_tool','show_shopping_form','task_dispatch'} <= names


async def test_extra_top_level_argument_is_rejected_before_execution():
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, ToolMessage
    from tests.test_agent_handoff import ScriptedModel, call
    executed=[]
    async def read(value: str):
        """合成只读工具。"""
        executed.append(value)
        return value
    tool=as_langchain_tool(read)
    model=ScriptedModel(responses=[call('read',{'value':'bad','invented':'not allowed'}),
                                   call('read',{'value':'ok'}),AIMessage(content='完成')])
    result=await create_agent(model,tools=[tool]).ainvoke({'messages':[{'role':'user','content':'测试'}]})
    receipts=[m for m in result['messages'] if isinstance(m,ToolMessage)]
    assert receipts[0].status=='error' and 'invented' in receipts[0].content
    assert receipts[1].status=='success' and executed==['ok']
