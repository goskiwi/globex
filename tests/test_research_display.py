"""研究展示的发布边界及持久恢复，不把候选变成最终推荐。"""
import pytest
from ag_ui.core import RunAgentInput
from app.application.agents.ag_ui_adapter import AGUIRunAdapter
from app.infrastructure.ag_ui_journal import AGUIJournal
from tests.test_ag_ui_journal import body
from tests.test_execution_process import start, end


@pytest.mark.parametrize('status', ['completed','partial'])
async def test_research_is_committed_only_after_final_delivery_and_restored_on_its_own_turn(tmp_path,status):
    journal = AGUIJournal(tmp_path/'research.db')
    request,_ = await journal.reserve(body('r1'),'b1','owner')
    events=[]
    adapter=AGUIRunAdapter(RunAgentInput.model_validate(request['input']),events.append)
    adapter.start()
    data={'guidance':'本次研究结果','reports':[{'task_id':'bag','items':[]}],
          'task_ids':['bag'],'completed_task_ids':['bag'] if status=='completed' else [],'status':status}
    start(adapter,'report','present_research')
    end(adapter,'report','present_research',data)
    assert not adapter.state['researchReports']
    adapter.finish(data['guidance'],status,None,delivery={"kind":"research","status":status})
    assert adapter.state['recommendation'] is None and adapter.state['comparison'] is None
    assert adapter.state['researchReports'][0]['runId']=='r1'
    await journal.append('r1','owner',[e.model_dump(mode='json',by_alias=True,exclude_none=True) for e in events])
    # 后续普通问句不抢走旧报告归属；重新打开日志后仍在原回复下。
    await journal.reserve(body('r2'),'b1','other')
    await journal.append('r2','other',[{'type':'RUN_FINISHED','threadId':'s1','runId':'r2'}])
    restored=await AGUIJournal(tmp_path/'research.db').session('s1','b1')
    assert restored['researchReports']==adapter.state['researchReports']
    assert restored['researchReports'][0]['status']==status


@pytest.mark.parametrize('ending',['unconfirmed','failed','cancelled'])
def test_prepared_research_is_not_published_on_failed_or_unconfirmed_delivery(ending):
    adapter=AGUIRunAdapter(RunAgentInput.model_validate(body()),lambda _:None)
    start(adapter,'report','present_research')
    end(adapter,'report','present_research',{'reports':[]})
    if ending=='cancelled':adapter.fail('已停止',cancelled=True)
    else:adapter.finish('尚未交付','failed' if ending=='failed' else 'completed',None)
    assert adapter.state['researchReports']==[]


def test_research_evidence_keeps_registered_media_without_changing_sku_facts():
    from app.application.runtime.handoff import TaskEvidence
    evidence=TaskEvidence('s')
    evidence._products([{'product_id':'P1003','title':'目录商品','image_url':'/products/wanderlite.png',
        'image_kind':'illustration','image_alt':'目录示意图','skus':[{'sku_id':'P1003-S1','price_major':129,'currency':'CNY','stock':2}]}],'ctx')
    facts=evidence.facts[('P1003','P1003-S1')]
    assert facts['product']['image_url']=='/products/wanderlite.png'
    assert facts['sku']['price_major']==129 and 'price_major' not in facts['product']


async def test_media_registry_exposes_only_explicit_product_and_asset_mapping():
    import httpx
    from fastapi import FastAPI
    from app.presentation.product_media import register_product_media_routes
    app=FastAPI();register_product_media_routes(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        response=await client.get('/commerce/product-media')
    assert response.status_code==200
    data=response.json()
    assert data['kind']=='illustration' and 'P30019' in data['products']
    assert data['assets'][data['products']['P30019']]['url'].startswith('/products/')
