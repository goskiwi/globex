"""过程摘要来自原生回执；关联、终态、并行和历史恢复不依赖关键词或模型正文。"""
import json
from types import SimpleNamespace

import pytest
from ag_ui.core import RunAgentInput

from app.application.agents.ag_ui_adapter import AGUIRunAdapter
from app.application.agents.execution_summary import result_summary
from app.application.runtime.events import ToolEvent
from app.infrastructure.ag_ui_journal import AGUIJournal
from app.infrastructure.eventbus import TradeEvent
from tests.test_ag_ui_journal import body


def adapter():
    return AGUIRunAdapter(RunAgentInput.model_validate(body('r')),lambda event:None)


def start(a,identifier,name,args=None):
    a.on_agent_event(ToolEvent('TOOL_CALL_START',identifier,name,arguments=args))
    a.on_agent_event(ToolEvent('TOOL_CALL_END',identifier,name))
    a.on_agent_event(ToolEvent('TOOL_RESULT_START',identifier,name))


def end(a,identifier,name,data=None,state='success'):
    a.on_agent_event(ToolEvent('TOOL_RESULT_END',identifier,name,state=state,data=data))


def test_parameters_ready_are_not_success_and_same_names_update_by_call_identity():
    a=adapter()
    for identifier in ('a','b'):
        start(a,identifier,'product_search_tool',{'normalized_query':'背包'})
    assert [s['status'] for s in a.state['process']['steps']]==['running','running']
    end(a,'b','product_search_tool',{'hits':[{'product_id':'P1001'}]})
    end(a,'a','product_search_tool',{'hits':[]})
    assert len(a.state['process']['steps'])==2
    assert '0个候选' in a.state['process']['steps'][0]['summary']
    assert '1个候选' in a.state['process']['steps'][1]['summary']
    assert 'progress' not in a.state


def test_condition_summary_uses_successful_structured_arguments_not_buyer_quote():
    args={'update':{'quote':'private buyer text','filters':{'landed_budget_major':300,'target_currency':'CNY','ship_to':'CN'}}}
    value=result_summary('update_shopping_state',args,'更新成功',True)
    assert '300 CNY' in value and '中国' in value and 'private buyer text' not in value
    assert '已满足' not in value


def test_details_report_existing_facts_not_invented_properties_or_catalog_ids():
    a=adapter();start(a,'d','get_product_details',{'product_id':'P1001'})
    end(a,'d','get_product_details',{'hits':[{'product_id':'P1001','title':'通勤背包',
        'source_platform':'amazon','highlights':['35L','可折叠'],'skus':[{'stock':0},{'stock':10}]}]})
    entry=a.state['process']['steps'][0]
    assert '通勤背包' in entry['label']
    assert '35L' in entry['summary'] and '2个规格' in entry['summary']
    assert 'P1001' not in json.dumps(entry,ensure_ascii=False)
    assert '背负' not in entry['summary']


def test_quote_total_is_grouped_purchase_not_three_individual_alternative_prices():
    data={'quote':{'items':[{'title':'背包'},{'title':'耳机'}],'ship_to':'CN','currency':'CNY','total_amount_minor':47800}}
    summary=result_summary('quote_products',{},data,True)
    assert '一起购买' in summary and '组合到手价 ¥478.00' in summary
    assert '中国' in summary


def test_child_partial_and_needs_input_do_not_become_completed_tool_success():
    for state,expected in [('partial','partial'),('needs_input','waiting_input')]:
        a=adapter();start(a,'task','task_dispatch',{'subagent_type':'search_agent'})
        end(a,'task','task_dispatch',{'agent':'search_agent','status':state,'candidates':[]})
        step=a.state['process']['steps'][0]
        assert step['label']=='商品研究' and step['status']==expected


@pytest.mark.parametrize('status,completed,expected', [
    ('completed',['a','b'],'completed'), ('partial',['a'],'partial')])
async def test_research_delivery_status_and_summary_survive_history_restore(tmp_path,status,completed,expected):
    journal=AGUIJournal(tmp_path/'runs.db')
    saved,_=await journal.reserve(body('r'),'b1','owner')
    events=[]
    target=AGUIRunAdapter(RunAgentInput.model_validate(saved['input']),events.append)
    target.start()
    start(target,'report','present_research',{'task_ids':['a','b']})
    end(target,'report','present_research',{'status':status,'task_ids':['a','b'],'completed_task_ids':completed})
    target.finish('研究报告',status,None)
    await journal.append('r','owner',[e.model_dump(mode='json',by_alias=True,exclude_none=True) for e in events])
    restored=await AGUIJournal(tmp_path/'runs.db').session('s1','b1')
    step=restored['run']['state']['process']['steps'][0]
    assert step['label']=='交付研究报告' and step['status']==expected
    assert '2份' in step['summary']
    assert ('未完成事项' in step['summary']) == (status=='partial')
    assert 'task_ids' not in json.dumps(step,ensure_ascii=False)


def test_type_change_summary_and_clear_are_explicit():
    assert result_summary('update_shopping_state',{'update':{'filters':{'product_type':'耳机'}}},{},True)=='商品类型：耳机'
    assert result_summary('update_shopping_state',{'update':{'filters':{'product_type':None}}},{},True)=='清除商品类型限制'


def test_pending_delivery_does_not_claim_published_before_commit():
    from tests.presentation_helpers import accept
    payload={'hits':[{'product_id':'P1001','title':'背包','default_sku_id':'P1001-S1'}],'preferred_sku_id':'P1001-S1'}
    a=adapter();start(a,'rec','recommend_products');end(a,'rec','recommend_products',payload)
    assert a.state['presentations']==[]
    assert a.state['process']['steps'][0]['status']=='unconfirmed'
    assert '已装配' in a.state['process']['steps'][0]['summary']
    accept(a,'recommendation',payload,identifier='rec')
    end(a,'rec','recommend_products',payload)
    a.finish('建议','completed',None)
    assert '已交付' in a.state['process']['steps'][0]['summary']
    assert a.state['process']['steps'][0]['status']=='completed'



@pytest.mark.parametrize('cancelled,status',[ (True,'cancelled'),(False,'failed') ])
def test_end_closes_only_unfinished_steps_without_fabricating_success(cancelled,status):
    a=adapter();start(a,'done','get_product_details');end(a,'done','get_product_details',{'hits':[]})
    start(a,'pending','quote_products')
    a.fail('结束',cancelled=cancelled)
    assert a.state['process']['status']==status
    assert a.state['process']['steps'][0]['status']=='completed'
    assert a.state['process']['steps'][1]['status']!='completed'


def test_failure_does_not_publish_address_secret_or_raw_exception():
    value=result_summary('create_order_tool',{'shipping_address':{'address_line':'private address'}},
        {'exception':'credential secret'},False)
    assert 'private address' not in value and 'credential secret' not in value
    assert '未确认' in value


def test_bad_model_arguments_do_not_break_progress_before_tool_validation():
    a=adapter();start(a,'bad','task_dispatch',{'subagent_type':[]})
    end(a,'bad','task_dispatch',{'error':'validation'},'error')
    assert a.state['process']['steps'][0]['status']=='failed'


async def test_journal_restores_each_process_under_its_own_user_turn(tmp_path):
    journal=AGUIJournal(tmp_path/'runs.db')
    for run_id,user_id in [('one','u-one'),('two','u-two')]:
        data=body(run_id);data['messages']=[{'id':user_id,'role':'user','content':'合成请求'}]
        await journal.reserve(data,'b1','owner')
        process={'runId':run_id,'userMessageId':user_id,'status':'completed','steps':[
            {'id':run_id+':call','label':'检索商品','summary':'返回2个候选','status':'completed'}]}
        await journal.append(run_id,'owner',[{'type':'STATE_SNAPSHOT','snapshot':{'process':process}},
            {'type':'RUN_FINISHED','threadId':'s1','runId':run_id}])
    restored=await journal.session('s1','b1')
    assert [(p['runId'],p['userMessageId']) for p in restored['processes']]==[('one','u-one'),('two','u-two')]
    assert all('2个候选' in p['steps'][0]['summary'] for p in restored['processes'])
