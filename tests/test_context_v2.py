from tests.shopping_state_helpers import work_fixture
"""历史字段关联、原生请求无损共享与来源校验。"""
import copy
import json
import pytest
from langchain_core.messages import ToolMessage,AIMessage
from app.application.runtime.results import ToolResultState
from app.infrastructure.context import ShoppingContext,ShoppingContextSnapshot
from app.infrastructure.context_products import business_view,product_page
from app.application.runtime.projections import read_output,share_identical_products
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup
from tests.native_context_helpers import history,policy

def product():
    return {'product_id': 'P887', 'description': '完整材质与限制：真皮部分不可水洗；合成面料允许擦拭；运费另计。' * 8,
            'skus': [{'sku_id': 'P887-S1', 'spec': '黑', 'price_major': 213, 'currency': 'CNY', 'stock': 37},
                     {'sku_id': 'P887-S2', 'spec': '蓝', 'price_major': 229, 'currency': 'CNY', 'stock': 19}]}

def test_description_only_exact_whole_long_repeats_removed():
    text = '完整材质与限制：真皮部分不可水洗；合成面料允许擦拭；运费另计，退货需原包装。'
    assert business_view({'description': text * 8})['description'] == text
    changed = text * 7 + text.replace('允许擦拭', '禁止擦拭')
    assert business_view({'description': changed})['description'] == changed
    assert business_view({'description': '哈哈' * 100})['description'] == '哈哈' * 100

@pytest.mark.parametrize('fields', ['skus', 'stock', 'sku_id,stock', 'price_major,currency', 'price',
                                   'price,stock', 'price_major,stock_quantity', 'unit_price,inventory'])
def test_lookup_fields_retain_sku_value_association(fields):
    page = product_page({'hits': [product()], 'observed_at': '2026-08-01'}, sku_id='P887-S2', fields=fields)
    assert page['hits'][0]['skus'] == [product()['skus'][1]]
    assert page['historical'] and page['observed_at'] == '2026-08-01'

def test_nondefault_sku_does_not_inherit_default_quote():
    hit = {**product(), 'default_sku_id': 'P887-S1', 'price_major': 213, 'currency': 'CNY',
           'landed_price': {'total_major': 238, 'shipping_major': 25}}
    page = product_page({'hits': [hit]}, sku_id='P887-S2', fields='price')
    quote = page['hits'][0]
    assert quote['sku_id'] == 'P887-S2' and quote['price_major'] == 229
    assert 'landed_price' not in quote
    original = product_page({'hits': [hit]}, sku_id='P887-S1', fields='price')['hits'][0]
    assert original['landed_price']['total_major'] == 238
    full = product_page({'hits': [hit]}, sku_id='P887-S2')['hits'][0]
    assert full['other_or_unspecified_sku_quote']['sku_id'] == 'P887-S1'
    assert hit['price_major'] == 213

async def test_lookup_filters_before_latest_and_never_crosses_buyer(tmp_path):
    store = ContextEvidenceStore(tmp_path / 'e.db')
    await store.save('buyer', 'session', 'display_batch', {'hits': [product()]})
    await store.save('buyer', 'session', 'display_batch', {'hits': [{'product_id': 'P999'}]})
    await store.save('other', 'session', 'display_batch', {'hits': [{**product(), 'secret': 'other'}]})
    token = ShoppingContext.set(ShoppingContextSnapshot('session', 'buyer', 'zh-CN', 'CNY'))
    try:
        result = await build_conversation_fact_lookup(store)(sku_id='P887-S2', fields='sku_id,stock')
        payload = json.loads(result.text)
        assert payload['records'][0]['data']['hits'][0]['skus'][0]['stock'] == 19
        assert payload['records'][0]['observation_scope']['time_basis'] == 'historical'
        assert 'secret' not in json.dumps(payload)
        missing = await build_conversation_fact_lookup(store)(batch=2, sku_id='P887-S2')
        assert json.loads(missing.text)['records'][0]['data']['hits'] == []
    finally:
        ShoppingContext.reset(token)

def test_shared_objects_round_trip_without_changing_order_or_quote_scope():
    messages = []
    for i, country in enumerate(['CN', 'CN', 'JP', 'CN']):
        hit = product()
        if i == 3:
            hit['skus'][0]['price_major'] = 217
        payload = {'hits': [hit], 'query_conditions': {'ship_to': country, 'currency': 'CNY', 'quantity': 1},
                   'result_ref': f'ctx_{i}', 'observed_at': f'2026-08-0{i+1}'}
        messages.append(ToolMessage(tool_call_id=f'call{i}',name='product_search_tool',content=json.dumps(payload), artifact={"data":payload}))
    original = copy.deepcopy(messages)
    shared = share_identical_products(messages)
    originals = {b.tool_call_id: read_output(b) for b in original}
    outputs = {b.tool_call_id: read_output(b) for b in shared}
    assert 'same_business_fields_as' in outputs['call1']['hits'][0]
    assert 'same_business_fields_as' not in outputs['call2']['hits'][0]
    assert 'same_business_fields_as' not in outputs['call3']['hits'][0]
    for call, payload in outputs.items():
        restored = copy.deepcopy(payload)
        restored.pop('shared_fields_notice', None)
        for i, hit in enumerate(restored['hits']):
            if 'same_business_fields_as' in hit:
                ref = hit['same_business_fields_as']
                restored['hits'][i] = outputs[ref['tool_call_id']]['hits'][ref['position'] - 1]
        assert restored == originals[call]
    assert [m.model_dump() for m in messages] == [m.model_dump() for m in original]
    # 去掉旧消息重新准备时不能产生悬空引用。
    assert 'same_business_fields_as' not in read_output(share_identical_products(messages[1:])[0])['hits'][0]

async def test_summary_sharing_never_references_outside_its_input():
    state=history()
    for m in state['messages']:
        if isinstance(m,ToolMessage):
            m.content=json.dumps({'hits':[product()]})
            m.artifact={'data':json.loads(m.content)}
    original=copy.deepcopy(state)
    head=share_identical_products(state['messages'][:15])
    ids={m.tool_call_id for m in head if isinstance(m,ToolMessage)}
    refs=[h['same_business_fields_as'] for m in head if isinstance(m,ToolMessage)
          for h in read_output(m)['hits'] if 'same_business_fields_as' in h]
    assert refs and all(r['tool_call_id'] in ids for r in refs)
    assert state==original


@pytest.mark.parametrize('first_score,second_score', [(0.9,0.2),(0.9,0),(None,0.2),(0.9,None)])
def test_query_score_does_not_duplicate_business_facts_or_inherit_old_score(first_score,second_score):
    messages=[]
    for i,score in enumerate((first_score,second_score)):
        hit=product()
        if score is not None:hit['score']=score
        payload={'hits':[hit],'query_conditions':{'normalized_query':f'query{i}','ship_to':'CN'},
                 'observed_at':f'time{i}','result_ref':f'ctx_{i}'}
        messages.append(ToolMessage(tool_call_id=f'call{i}',name='product_search_tool',
                                    content=json.dumps(payload),artifact={'data':payload}))
    original=copy.deepcopy(messages)
    projected=share_identical_products(messages)
    source=read_output(projected[0])['hits'][0]
    second=read_output(projected[1])
    shared=second['hits'][0]
    assert shared['same_business_fields_as']=={'tool_call_id':'call0','position':1}
    assert ('score' in shared)==(second_score is not None)
    if second_score is not None:assert shared['score']==second_score
    restored={k:v for k,v in source.items() if k!='score'}
    restored.update({k:v for k,v in shared.items() if k!='same_business_fields_as'})
    assert restored==read_output(original[1])['hits'][0]
    assert second['query_conditions']['normalized_query']=='query1' and second['observed_at']=='time1'
    assert messages==original


@pytest.mark.parametrize('field,value', [('weight_kg',0.7),('rating_summary',{'review_count':50})])
def test_changed_product_facts_are_not_shared_when_scores_differ(field,value):
    first={**product(),field:value,'score':0.9}
    second={**first,field:None,'score':0.2}
    messages=[ToolMessage(tool_call_id=f'call{i}',content=json.dumps({'hits':[hit]}),
                          artifact={'data':{'hits':[hit]}}) for i,hit in enumerate((first,second))]
    assert 'same_business_fields_as' not in read_output(share_identical_products(messages)[1])['hits'][0]


@pytest.mark.parametrize('tool,nested', [('product_search_tool',False),('recommend_products',False),
                                      ('compare_products',False),('conversation_fact_lookup',True)])
def test_catalog_admin_fields_stay_in_evidence_not_model_decisions(tool,nested):
    hit={**product(),'data_provenance':'synthetic','image_kind':'illustration',
         'image_url':'/products/example.png','weight_kg':0.38,'rating_summary':{'review_count':52}}
    data={'records':[{'data':{'hits':[hit]}}]} if nested else {'hits':[hit]}
    message=ToolMessage(tool_call_id='call',name=tool,content=json.dumps(data),artifact={'data':data})
    original=copy.deepcopy(message)
    projected=share_identical_products([message])[0]
    model_data=read_output(projected)
    selected=model_data['records'][0]['data']['hits'][0] if nested else model_data['hits'][0]
    assert 'data_provenance' not in selected and 'image_kind' not in selected and 'image_url' not in selected
    assert selected['weight_kg']==0.38 and selected['rating_summary']=={'review_count':52}
    assert selected['skus']==hit['skus'] and selected['description']==hit['description']
    assert projected.artifact['data']==original.artifact['data']
    assert message==original


def test_compact_projection_preserves_harness_notices():
    from app.application.runtime.results import project_data
    message=project_data(ToolMessage(tool_call_id='read',content=''),{'hits':[product()]},notices=['不得据此编造数据'])
    projected=share_identical_products([message],compact_rules=True)[0]
    assert projected.artifact['notices']==['不得据此编造数据']
    assert json.loads(projected.content)['notices']==['不得据此编造数据']

async def test_summary_repair_only_reads_original_sources_and_rejected_is_not_recalled(tmp_path):
    from tests.native_context_helpers import summary_selection
    from app.application.runtime.context_summary import render_summary
    state=history();mw=policy(tmp_path,target_tokens=10)
    mw.model.ainvoke.side_effect=[AIMessage(content='错误 P999999-S1'),AIMessage(content=summary_selection(1))]
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    try:
        update=await mw.compact_checkpoint(state,force=True)
        assert 'P0-S1' in render_summary(update['context_summary']) and 'P999999' not in render_summary(update['context_summary'])
        assert [x['status'] for x in update['context_statistics']['summary_attempts']]==['rejected','accepted']
        calls=mw.model.ainvoke.call_args_list
        assert calls[0].args[0][1].content==calls[1].args[0][1].content
        assert not await mw.store.search('b','s',query='P999999')
        ref=await mw.store.save('b','s','rejected_summary',{'candidate':'P999999'})
        assert (await build_conversation_fact_lookup(mw.store)(result_ref=ref)).state==ToolResultState.ERROR
    finally:ShoppingContext.reset(token)

async def test_archived_product_and_current_working_are_valid_summary_sources(tmp_path):
    from tests.native_context_helpers import summary_selection
    from app.application.runtime.context_summary import render_summary
    mw=policy(tmp_path,target_tokens=10,response=summary_selection(1))
    state=history()
    ref=await mw.store.save('b','s','products',{'hits':[{'product_id':'P8483','skus':[{'sku_id':'P8483-S1','stock':72}]}]})
    state['messages'][2].content=json.dumps({'archived':True,'result_ref':ref})
    state['messages'][2].artifact={'data':{'archived':True,'result_ref':ref}}
    state['shopping_work']=work_fixture(selected=['P8483-S1'],budget=352)
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    try:
        update = await mw.compact_checkpoint(state,force=True)
        assert ref in render_summary(update['context_summary'])
        assert state['shopping_work']['selections']['P8483-S1']['sku_id'] == 'P8483-S1'
        assert '"stock": 72' not in mw.model.ainvoke.call_args.args[0][1].content
    finally:ShoppingContext.reset(token)
