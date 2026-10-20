"""审查复现：显式要求已由专家核对后，计划是否仍被声明时的未知状态永久阻塞。"""
import json
from langchain_core.messages import AIMessage, ToolMessage
from tests.test_agent_handoff import ScriptedModel, call, submission
from tests.test_langgraph_runtime import _container
from tests.test_task_execution import ask

async def test_verified_requirement_can_complete_delivery(tmp_path,monkeypatch):
    child=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P3008','sku_id':'P3008-S1'}),
        submission(candidates=[{'product_id':'P3008','sku_id':'P3008-S1','assessment':'目录明确记载配防雨罩，符合本次明确要求。'}])])
    model=ScriptedModel(responses=[call('set_research_plan',{'plan': [{
        'id':'bag','goal':'找一款配防雨罩的背包','filters':{'product_type':'背包','price_max_major':300,'target_currency':'CNY','ship_to':'CN'},
        'requirements':['必须明确配防雨罩']} ]}),
        call('task_dispatch',{'subagent_type':'search_agent','task':{'goal':'按计划研究','step_id':'bag'}}),
        call('recommend_products', {'picks': [{'product_id': 'P3008', 'sku_id': 'P3008-S1', 'quantity': 1, 'reason': '目录配防雨罩', 'task_id': 'bag'}], 'guidance': '这款目录明确配防雨罩，符合明确需求。'}, finish=True)])
    c=await _container(tmp_path,monkeypatch,model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    try:
        result,state=await ask(c,'请按计划找一款明确配防雨罩的背包并推荐。')
        dispatch=next(m.artifact['data'] for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='task_dispatch')
        print(json.dumps({'final_status':result.status,'final_text':result.final_text,'child_status':dispatch['status'],
            'child_unmet':dispatch['unmet_requirements'],'read_description':dispatch['candidates'][0]['facts']['product']['description'],
            'qualified_skus':dispatch['qualified_skus'],'task':state['task_plan']['tasks']['bag'],'steps':state['task_plan']['steps']},ensure_ascii=False))
        assert dispatch['status']=='completed' and not dispatch['unmet_requirements']
        assert '配防雨罩' in dispatch['candidates'][0]['facts']['product']['description']
        assert result.status=='completed','已确认要求仍被永久视作未知，计划无法完成'
    finally:await c.shutdown()
