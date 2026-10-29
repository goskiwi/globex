"""真实API、SDK与图执行验证改进链；不以伪造优化分数代替业务行为。"""
import json
import asyncio
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import yaml

from scripts.eval.task_evaluator import TaskEvaluator
from scripts.eval.trace import tool_observations
from scripts.eval.evidence import evaluate_trace_assertions
from scripts.improvement.experience import contrast_records, experience_documents
from tests.native_model_helpers import client_model, completion
from tests.test_retrieval import _settings
from tests.test_capability_registry import document as skill_document


def task_case(identifier='exact-sku', country=None):
    expected={'sku_id':'P1003-S1'}
    if country: expected['ship_to']=country
    assertions=[{'criterion':'搜索真正成功且条件一致','kind':'search_conditions','expected':expected},
                {'criterion':'交付','kind':'answer_present'}, {'criterion':'无运行错误','kind':'no_execution_error'}]
    return {'id':identifier,'split':'holdout','family':identifier,'description':'合成精确查询',
            'queries':['只查 P1003-S1，不下单。'],
            'rubric':{'p0':[a['criterion'] for a in assertions],'p1':[],'p2':[]},
            'deterministic':{'p0':assertions,'p1':[],'p2':[]}}


def tool_reply(name,args,index):
    raw=completion()
    raw['choices'][0].update(finish_reason='tool_calls',message={'role':'assistant','content':None,
        'tool_calls':[{'id':f'call-{index}','type':'function',
                       'function':{'name':name,'arguments':json.dumps(args)}}]})
    return raw


@pytest.mark.parametrize('role,country,expected_score',[('main',None,1.0),('search',None,0.0),('main','ZZ',0.0)])
async def test_actual_prompt_assignment_role_and_successful_business_result(tmp_path,role,country,expected_score):
    requests=[]
    def handler(request):
        payload=json.loads(request.content); requests.append(payload)
        if len(requests)==1:
            args={'sku_id':'P1003-S1'}
            if country:args['conditions']={'filters':{'ship_to':country}}
            raw=tool_reply('product_search_tool',args,1)
        else:
            raw=completion();raw['choices'][0]['message']['content']='已按实际工具结果完成查询。'
        return httpx.Response(200,json=raw)
    model=await client_model(tmp_path,handler)
    document=yaml.safe_load(Path('app/application/prompts/globex.yml').read_text())
    evaluator=TaskEvaluator(_settings(tmp_path),None,document=document,output=tmp_path/'experiment',
                            role=role,model_factory=lambda *a,**k:model)
    try:
        result=await evaluator.run(task_case(country=country),'CANDIDATE_ROLE_SENTINEL')
        raw=json.loads((evaluator.output/result['execution_ref']).read_text())
        assert result['score']==expected_score,raw
        assert raw['candidate_execution']['version_executed']
        assert raw['candidate_execution']['target_executed']==(role=='main')
        if role=='main': assert 'CANDIDATE_ROLE_SENTINEL' in str(requests[0])
        else: assert 'CANDIDATE_ROLE_SENTINEL' not in str(requests[0])
        calls=tool_observations(raw['trace_events'])
        search=next(row for row in calls if row['name']=='product_search_tool')
        assert search['status']==('error' if country else 'success')
        assert evaluator.settings.data_dir==tmp_path
    finally:
        await evaluator.close();await model.aclose()


async def test_skill_body_is_used_only_after_real_catalog_load(tmp_path):
    source=skill_document()
    source.update(id='test-experience',version='trial-1',allowed_tools=[],body='ORIGINAL_BODY')
    requests=[]
    def handler(request):
        payload=json.loads(request.content); requests.append(payload)
        if len(requests)==1:
            assert 'test-experience' in str(payload)
            assert 'EXPERIENCE_BODY_SENTINEL' not in str(payload)
            raw=tool_reply('load_agent_skill_tool',{'skill_id':source['id'],'version':source['version']},1)
        else:
            assert 'EXPERIENCE_BODY_SENTINEL' in str(payload)
            raw=completion();raw['choices'][0]['message']['content']='已读取流程，按其适用范围回答。'
        return httpx.Response(200,json=raw)
    model=await client_model(tmp_path,handler)
    document=yaml.safe_load(Path('app/application/prompts/globex.yml').read_text())
    evaluator=TaskEvaluator(_settings(tmp_path),None,document=document,output=tmp_path/'trial',
        role='main',skill_source=source,model_factory=lambda *a,**k:model)
    case=task_case()
    case['rubric']={'p0':['加载成功','交付'],'p1':[],'p2':[]}
    case['deterministic']={'p0':[{'criterion':'加载成功','kind':'skill_loaded'},
                               {'criterion':'交付','kind':'answer_present'}],'p1':[],'p2':[]}
    try:
        result=await evaluator.run(case,'EXPERIENCE_BODY_SENTINEL')
        assert result['score']==1.0,result
        assert len(requests)==2
        assert not (tmp_path/'capabilities.db').exists()
    finally:
        await evaluator.close();await model.aclose()


async def test_search_candidate_runs_through_parent_child_acceptance_and_public_delivery(tmp_path):
    requests=[]
    def handler(request):
        payload=json.loads(request.content); requests.append(payload)
        index=len(requests)
        if index==1:
            raw=tool_reply('task_dispatch',{'subagent_type':'search_agent',
                'task':{'goal':'核对P1003-S1并交回已读证据'}},index)
        elif index==2:
            assert 'SEARCH_CANDIDATE_SENTINEL' in str(payload)
            assert 'delegated_task' in str(payload)
            raw=tool_reply('product_search_tool',{'sku_id':'P1003-S1'},index)
        elif index==3:
            raw=tool_reply('SubagentResult',{'candidates':[{'product_id':'P1003','sku_id':'P1003-S1',
                'evidence_fields':['description','weight_kg']}]},index)
        elif index==4:
            receipt=next(json.loads(message['content']) for message in payload['messages']
                         if message.get('role')=='tool' and '"agent": "search_agent"' in message['content'])
            raw=tool_reply('present_research',{'reports':[{'task_id':receipt['scope']['task_id'],
                'title':'合成规格研究'}],'turn_action':'finish'},index)
        else:
            raise AssertionError('交付后不应再请求模型')
        return httpx.Response(200,json=raw)
    model=await client_model(tmp_path,handler)
    document=yaml.safe_load(Path('app/application/prompts/globex.yml').read_text())
    evaluator=TaskEvaluator(_settings(tmp_path),None,document=document,output=tmp_path/'trial',
                            role='search',model_factory=lambda *a,**k:model)
    case=task_case()
    case['rubric']['p0'].append('研究接纳并公开交付')
    case['deterministic']['p0'].append({'criterion':'研究接纳并公开交付','kind':'delegation_delivered','minimum':1})
    try:
        result=await evaluator.run(case,'SEARCH_CANDIDATE_SENTINEL')
        assert result['score']==1.0,result
        assert result['feedback']['candidate_execution']['target_executed']
        assert len(requests)==4
    finally:
        await evaluator.close();await model.aclose()


def test_experience_uses_real_variant_contrasts_and_never_holdout():
    records=[{'id':'one','split':'train','candidate_hash':variant,'execution_ref':f'executions/{i}/result.json','score':i}
             for i,variant in enumerate(['old','new'])]
    records.append({**records[0],'split':'holdout','id':'unseen'})
    eligible=contrast_records(records)
    assert len(eligible)==2
    item={'title':'按证据收尾','description':'目录证据充分时使用','scope':'shopping',
          'cue':'明确目标的研究','lesson':'交付已读候选和字段缺口','limitations':'不能替代买家约束',
          'evidence_indices':[0,1]}
    documents=experience_documents(json.dumps({'experiences':[item]}),eligible,version='trial',reference_root=Path('runs/one'))
    assert documents[0]['kind']=='skill'
    assert len(documents[0]['evidence'])==2
    with pytest.raises(ValueError,match='不同候选'):
        experience_documents(json.dumps({'experiences':[{**item,'evidence_indices':[0]}]}),eligible,
                             version='trial',reference_root=Path('runs/one'))
    with pytest.raises(ValueError,match='不存在'):
        experience_documents(json.dumps({'experiences':[{**item,'evidence_indices':[0,2]}]}),eligible,
                             version='trial',reference_root=Path('runs/one'))


def test_gepa_to_evidence_experience_and_new_session_reuse_through_real_graphs(tmp_path):
    """真实优化引擎和应用执行，模型HTTP输出为脚本；不代表外部模型效果。"""
    from scripts.optimize_prompts import optimize_candidate
    from scripts.improvement.experience import distill_experience, verify_reuse
    requests=[]
    def handler(request):
        payload=json.loads(request.content);requests.append(payload)
        system=str(payload['messages'][0]['content'])
        raw=completion()
        if '离线执行经验分析员' in system:
            row={'title':'精确规格研究','description':'核对明确SKU时使用','scope':'shopping',
                 'cue':'明确指定规格','lesson':'USE_CORRECT_SKU：使用独立sku_id精确查询，保留指定规格',
                 'limitations':'不改变买家约束或执行交易','evidence_indices':[0,1]}
            raw['choices'][0]['message']['content']=json.dumps({'experiences':[row]},ensure_ascii=False)
        elif 'execution-experience-1' in str(payload) and 'USE_CORRECT_SKU' not in str(payload):
            raw=tool_reply('load_agent_skill_tool',{'skill_id':'execution-experience-1','version':'trial'},len(requests))
        elif not any(message.get('role')=='tool' and 'query_conditions' in str(message.get('content'))
                     for message in payload['messages']):
            correct='GOOD_CANDIDATE' in system or 'USE_CORRECT_SKU' in str(payload)
            raw=tool_reply('product_search_tool',{'sku_id':'P1003-S1' if correct else 'P1001-S1'},len(requests))
        else:
            raw['choices'][0]['message']['content']='按实际查询结果完成。'
        return httpx.Response(200,json=raw)
    cases=[]
    for split in ['train','val','holdout']:
        case=task_case(split);case['split']=split;case['queries']=[split+'：核对P1003-S1']
        cases.append(case)
    document=yaml.safe_load(Path('app/application/prompts/globex.yml').read_text())
    with asyncio.Runner() as runner:
        model=runner.run(client_model(tmp_path,handler))
        evaluator=TaskEvaluator(_settings(tmp_path),runner,document=document,output=tmp_path/'loop',role='main',
                                model_factory=lambda *a,**k:model)
        evaluator.model=model
        try:
            report=optimize_candidate('BAD_CANDIDATE',cases,evaluator,evaluator.output,max_calls=8,
                proposer=lambda candidate,dataset,components:{'instruction':'GOOD_CANDIDATE'})
            assert report['holdout']['baseline_scores']==[0]
            assert report['holdout']['candidate_scores'][0]>0
            assert report['holdout']['task_outcomes']==[{'id':'holdout','baseline':0,'candidate':1}]
            documents=distill_experience(report['records'],evaluator.reflect,output=evaluator.output,version='trial')
            assert len(documents)==1
            eligible=json.loads((evaluator.output/'experience-input.json').read_text())
            assert all(row['split']!='holdout' for row in eligible)
            reuse=verify_reuse(cases,evaluator,'BAD_CANDIDATE',documents,evaluator.output)
            assert reuse[0]['without']['score']==0 and reuse[0]['with']['score']==1
            assert reuse[0]['actual_reuse'] and reuse[0]['business_passed']
            assert not list(evaluator.output.rglob('*.db'))
        finally:
            runner.run(evaluator.close())


def test_model_service_failure_preserves_execution_and_aborts_learning(tmp_path):
    from scripts.eval.task_evaluator import TaskEvaluationUnavailable
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(402,json={'error':{'message':'synthetic insufficient balance','type':'insufficient_balance_error'}})
    document=yaml.safe_load(Path('app/application/prompts/globex.yml').read_text())
    with asyncio.Runner() as runner:
        model=runner.run(client_model(tmp_path,handler))
        evaluator=TaskEvaluator(_settings(tmp_path),runner,document=document,output=tmp_path/'unavailable',role='main',
                                model_factory=lambda *a,**k:model)
        try:
            with pytest.raises(TaskEvaluationUnavailable) as error:
                evaluator(task_case(),'SERVICE_FAILURE_SENTINEL')
            result=json.loads((evaluator.output/error.value.execution_ref).read_text())
            assert result['candidate_execution']['version_executed']
            assert result['provider_failures'][0]['http_status']==402
            assert result['verdict']!='PASS'
            assert len(requests)==1
        finally:
            runner.run(evaluator.close());runner.run(model.aclose())


async def test_cancelled_inflight_task_keeps_partial_evidence_and_unknown_usage(tmp_path):
    started=asyncio.Event()
    async def handler(request):
        started.set()
        await asyncio.Future()
    model=await client_model(tmp_path,handler)
    document=yaml.safe_load(Path('app/application/prompts/globex.yml').read_text())
    evaluator=TaskEvaluator(_settings(tmp_path),None,document=document,output=tmp_path/'cancelled',role='main',
                            model_factory=lambda *a,**k:model)
    try:
        task=asyncio.create_task(evaluator.run(task_case(),'CANCELLED_CANDIDATE'))
        await asyncio.wait_for(started.wait(),5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task,5)
        result=json.loads(next(evaluator.output.rglob('result.json')).read_text())
        assert result['verdict']=='INTERRUPTED'
        assert result['candidate_execution']['version_executed']
        assert result['model_calls']==1
        assert evaluator.usage_summary()['input_tokens'] is None
        assert evaluator.usage_summary()['cost_usd'] is None
    finally:
        await evaluator.close();await model.aclose()


def test_application_logs_outlive_gepa_temporary_output_streams(tmp_path):
    # 独立进程避开pytest已经安装的日志handler，覆盖CLI首次导入应用的真实时序。
    code='''
import logging
from scripts.eval.agent_runtime import isolated_runtime
from gepa.logging.logger import Logger
with Logger("run_log.txt"):
    logging.getLogger("httpx").info("inside-optimization")
logging.getLogger("httpx").info("after-optimization")
'''
    import os
    result=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,
        env={**os.environ,'PYTHONPATH':str(Path.cwd())},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'after-optimization' in result.stderr
    assert 'Logging error' not in result.stderr and 'closed file' not in result.stderr


async def test_positive_improvement_cases_have_real_catalogue_candidates():
    from app.application.usecases.catalog_search import CatalogSearchUseCase
    from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
    from app.domain.catalog.product_search_spec import ProductSearchSpec
    search=CatalogSearchUseCase(InMemoryProductRepository())
    rows=[json.loads(line) for line in Path('eval/optimization_search.jsonl').read_text().splitlines()]
    for row in rows:
        conditions=next(a['expected'] for a in row['deterministic']['p0'] if a['kind']=='search_conditions')
        result=await search.execute(ProductSearchSpec(normalized_query=row['catalogue_query'],
            top_k=10,**conditions))
        assert any(row['catalogue_query'] in hit.get('title','') for hit in result['hits']), \
            f"正向研究任务 {row['id']} 在实际目录和硬条件下没有对应候选"
