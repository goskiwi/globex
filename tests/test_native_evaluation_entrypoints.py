"""完整评测入口实际执行原生图、生产工具、SQLite；仅隔离模型 HTTP。"""
import json
import re
from dataclasses import replace
import httpx
import pytest
from tests.native_model_helpers import client_model,completion
from tests.test_retrieval import _settings


def scripted_response(request):
    body=json.loads(request.content);messages=body['messages'];raw=completion()
    buyers=[(i,m) for i,m in enumerate(messages) if m.get('name','').startswith('harness-') and m['role']=='user']
    if not buyers:
        from tests.native_context_helpers import summary_selection
        raw['choices'][0]['message']['content']=summary_selection()
        return httpx.Response(200,json=raw)
    index,buyer=buyers[-1];query=buyer['content']
    if not isinstance(query,str):query=''.join(x.get('text','') for x in query)
    results=[m for m in messages[index+1:] if m['role']=='tool']
    names=[m.get('name') for m in results]
    # OpenAI 工具消息可能省略 name，用同轮调用记录核对已执行工具。
    calls=[c['function']['name'] for m in messages[index+1:] for c in m.get('tool_calls',[])]
    tool=None;args={}
    if '拒绝' in query:pass
    elif '表单' in query and not results:
        tool='show_shopping_form';args={'title':'耳机用途','questions':[{'id':'use','type':'text','label':'主要用途？'}]}
    elif '确认卡' in query and '取消' not in query and 'create_order_tool' not in calls:
        if 'update_shopping_state' not in calls:
            tool='update_shopping_state';args={'update':{'selections':[{'product_id':'P1003','sku_id':'P1003-S1','quantity':1}]}}
        else:
            tool='create_order_tool';args={'sku_ids':['P1003-S1'],
                'shipping_address':{'recipient_name':'评测买家','country':'CN','state':'上海','city':'上海','address_line':'评测街1号','postal_code':'200000','phone':'00000000000'}}
    elif '订单 ' in query:
        order=re.search(r'订单\s+(\S+)',query).group(1)
        if not results:tool='query_order_tool';args={'order_id':order}
        elif '申请取消' in query and 'cancel_order_tool' not in calls:
            tool='cancel_order_tool';args={'order_id':order,'reason':'买家不再需要'}
    elif not results:tool='product_search_tool';args={'product_id':'P1003','sku_id':'P1003-S1'}
    if tool:
        raw['choices'][0]['finish_reason']='tool_calls'
        raw['choices'][0]['message']={'role':'assistant','content':None,'tool_calls':[
            {'id':'call-'+str(len(messages)),'type':'function','function':{'name':tool,'arguments':json.dumps(args,ensure_ascii=False)}}]}
    else:
        raw['choices'][0]['message']['content']=json.dumps({'sku_id':'P1003-S1','unit_price_major':129,'stock':80,'budget_major':180,'currency':'CNY'}) if '只用 JSON' in query else '已核对，未执行未经批准的操作。'
    return httpx.Response(200,json=raw)


@pytest.mark.parametrize('scenario',['sku','restart','injection','form','approval','cancel','owner','long'])
async def test_workflow_entrypoint_uses_production_graph_and_tools(tmp_path,monkeypatch,scenario):
    from scripts.eval.harness import workflows
    model=await client_model(tmp_path,scripted_response)
    monkeypatch.setattr(workflows,'create_chat_client',lambda settings:model.client)
    monkeypatch.setattr(workflows,'create_chat_model',lambda *args,**kwargs:model)
    row=await workflows.run_workflow({'id':scenario,'scenario':scenario,'mode':'long' if scenario=='long' else 'workflow'},
        replace(_settings(tmp_path),harness_enabled=True,llm_fallback_model=''),None,tmp_path/scenario,0)
    assert row['passed'],row
    assert row['tool_trace'] and row['usage']
    assert model.client.is_closed()


@pytest.mark.parametrize('mode',['snapshot','long'])
async def test_context_benchmark_uses_native_checkpoint(tmp_path,monkeypatch,mode):
    from scripts.eval import run_context
    def handler(request):
        body=json.loads(request.content);raw=completion()
        query=body['messages'][-1]['content']
        if '整理历史交接' in str(body['messages'][0]['content']):
            from tests.native_context_helpers import summary_selection
            raw['choices'][0]['message']['content']=summary_selection()
        elif isinstance(query,str) and '调用 product_search_tool' in query:
            number=int(re.search(r'第(\d+)批',query).group(1))
            raw['choices'][0]['finish_reason']='tool_calls'
            raw['choices'][0]['message']={'role':'assistant','content':None,'tool_calls':[{
                'id':f'batch-{number}','type':'function','function':{'name':'product_search_tool','arguments':json.dumps({'batch':number})}}]}
        else:raw['choices'][0]['message']['content']='已记录。'
        return httpx.Response(200,json=raw)
    model=await client_model(tmp_path,handler)
    monkeypatch.setattr(run_context,'create_chat_model',lambda *a,**k:model)
    case={'id':'native','split':'dev','mode':mode,'question':'总结需求','contains':['已记录'],'requires_current':False,
        'fixture':{'rounds':7,'compact_rounds':[4,7],'restart_round':5,'description_repetitions':1}}
    row=await run_context.Benchmark(_settings(tmp_path),None,tmp_path/'run').run_case(case,'layered',0)
    assert row['passed'],json.dumps({k:row[k] for k in ['error','checks','compactions']},ensure_ascii=False)
    assert row['model_calls']>0 and model.client.is_closed()
