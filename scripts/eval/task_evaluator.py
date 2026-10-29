"""优化候选直接运行生产 API、主子图、交付与审批；完整证据逐次保存。"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3
import tempfile

import httpx
import yaml

from app.infrastructure.llm import create_chat_client, create_chat_model
from scripts.eval.agent_runtime import capture_evaluation, isolated_runtime, retrieval_evidence
from scripts.eval.trace import tool_observations
from scripts import eval_regression


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class TaskEvaluationUnavailable(RuntimeError):
    """真实模型服务未完成请求；原始结果保留，不作为候选质量反馈。"""

    def __init__(self, execution_ref):
        self.execution_ref = execution_ref
        super().__init__('模型服务未完成请求，离线改进中止；见 ' + execution_ref)


class TaskEvaluator:
    """候选只替换已声明角色正文或公共 Skill，其余执行职责共用实际应用。"""

    def __init__(self, settings, runner, *, document, output, role='search', skill_source=None,
                 model_factory=None, isolated_knowledge=False):
        if role not in {'main', 'search', 'trade'}:
            raise ValueError('role 必须是 main、search 或 trade')
        self.settings, self.runner = settings, runner
        self.document, self.output, self.role = deepcopy(document), Path(output), role
        self.skill_source = deepcopy(skill_source)
        self.model_factory, self.isolated_knowledge = model_factory, isolated_knowledge
        self.reflection_usage, self.records = [], []
        self._sequence = 0
        self._client = create_chat_client(settings)
        self.model = create_chat_model(settings, stream=False, client=self._client)

    async def close(self):
        await self.model.aclose()
        await self._client.close()

    def usage_summary(self):
        metrics = [row['output'].get('metrics') or {} for row in self.records]
        def total(values):
            return sum(values) if values and all(type(v) in (int,float) and math.isfinite(v) and v >= 0
                                                 for v in values) else None
        def tokens(key):
            return total([row.get(key) for row in metrics] + [row.get(key) for row in self.reflection_usage])
        return {'task_executions':len(self.records),
            'task_model_requests':sum(row['output']['model_calls'] for row in self.records),
            'task_tool_calls':sum(row['output']['tool_calls'] for row in self.records),
            'reflection_model_requests':len(self.reflection_usage),
            'input_tokens':tokens('input_tokens'), 'output_tokens':tokens('output_tokens'),
            'cost_usd':total([row.get('cost_usd') for row in metrics]+
                [row.get('prompt_cache',{}).get('reported_cost') for row in self.reflection_usage]),
            'cost_scope':'全部任务执行（含留出与经验复用）及反思；缺失值保持未知，不按调用数推算费用'}

    def candidate_document(self, text):
        document = deepcopy(self.document)
        target = document['main_agent'] if self.role == 'main' else document['sub_agents'][self.role]
        target['system_prompt'] = text
        return document

    async def run(self, case, text, *, capabilities=(), skill_source=None):
        self._sequence += 1
        execution = self.output / 'executions' / f'{self._sequence:04d}-{case["id"]}'
        execution.mkdir(parents=True, exist_ok=False)
        source = skill_source if skill_source is not None else self.skill_source
        document = self.document if source is not None else self.candidate_document(text)
        prompt_path = execution / 'prompt.yml'
        prompt_path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False))
        installed = list(capabilities)
        if source is not None:
            installed.append({**deepcopy(source), 'body': text})
        events = []
        result = {'id': case['id'], 'score': 0.0, 'p0_pass': False, 'verdict': 'ERROR',
                  'judged': {}, 'trace_events': events, 'metrics': {}}
        assignment = []
        intended_version = None
        interruption = None
        try:
            with tempfile.TemporaryDirectory(prefix='globex-improvement-') as temporary:
                async with isolated_runtime(Path(temporary) / 'runtime', self.settings,
                        prompt_path=prompt_path, capabilities=installed,
                        model_factory=self.model_factory, isolated_knowledge=self.isolated_knowledge) as (app, container):
                    intended_version = container.prompt_registry.describe()['effective_version']['version_id']
                    try:
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                                    base_url=eval_regression.BASE_URL) as client:
                            with capture_evaluation(events):
                                async with asyncio.timeout(case.get('timeout_seconds', 240)):
                                    result = await eval_regression.run_case(client, case, '',
                                        identity_policy=app.state.identity_policy, evidence_events=events)
                    finally:
                        # 即使模型或业务失败，也保留已经发生的版本分配，不能误报成未选中候选。
                        with sqlite3.connect(container.prompt_registry.path) as db:
                            assignment = [dict(zip(('session_id', 'version_id'), row)) for row in db.execute(
                                'SELECT session_id,version_id FROM prompt_assignments')]
        except asyncio.CancelledError as error:
            interruption = error
            result.update(error='CancelledError',verdict='INTERRUPTED',score=0.0,p0_pass=False)
        except Exception as error:
            result.update(error=type(error).__name__, verdict='ERROR', score=0.0, p0_pass=False)
            if isinstance(error, eval_regression.CaseExecutionError):
                events = error.trace_events
                result['transcript'] = error.transcript
        result['trace_events'] = events
        provider_failures = [e['payload']['failure'] for e in events
            if e.get('type') == 'eval.model_response_contract' and e.get('payload',{}).get('failure')
            and (type(e['payload']['failure'].get('http_status')) is int
                 or e['payload']['failure']['type'] in {'APIConnectionError','APITimeoutError'})]
        result['provider_failures'] = provider_failures
        calls = tool_observations(events)
        target_executed = (self.role == 'main' or any(
            row['name'] == 'task_dispatch' and row['status'] == 'success'
            and row['data'].get('agent') == self.role + '_agent' for row in calls))
        if source is not None:
            target_executed = any(row['name'] == 'load_agent_skill_tool' and row['status'] == 'success'
                and row['data'].get('id') == source['id'] and row['data'].get('version') == source['version']
                for row in calls)
        version_executed = bool(assignment) and all(row['version_id'] == intended_version for row in assignment)
        result['candidate_execution'] = {'role': self.role, 'prompt_version': intended_version,
            'assignments': assignment, 'target_executed': target_executed, 'version_executed': version_executed,
            'skill': {key: source[key] for key in ('id','version')} if source else None}
        result['experience_reuse'] = [{'id': item['id'], 'version': item['version'],
            'loaded': any(row['name'] == 'load_agent_skill_tool' and row['status'] == 'success'
                and isinstance(row['data'],dict) and row['data'].get('id') == item['id']
                and row['data'].get('version') == item['version'] for row in calls)} for item in capabilities]
        if not target_executed or not version_executed:
            result.update(score=0.0, verdict='FAIL', p0_pass=False)
        result['retrieval'] = {**retrieval_evidence(events), 'configured':{
            'embedding_model':self.settings.embedding_model,
            'qdrant_collection':self.settings.qdrant_collection,
            'category_kb_collection':self.settings.category_kb_collection,
            'reranker_model':self.settings.reranker_model,
            'hybrid_recall_enabled':self.settings.hybrid_recall_enabled}}
        result['tool_failures'] = [{'name': row['name'], 'failure': row['failure']} for row in calls if row['status'] == 'error']
        result['tool_calls'] = len(calls)
        result['model_calls'] = sum(e.get('type') == 'eval.model_request' for e in events)
        result['execution_mode'] = 'production_api_and_graphs_isolated_state'
        result['manual_review'] = {'status':'PENDING', 'scope':'自由解释与经验因果归纳不由结构化断言证明'}
        write_json(execution / 'result.json', result)
        violations = [item['criterion'] for rows in result.get('judged', {}).values()
                      for item in rows if item.get('pass') is not True]
        if not target_executed: violations.append('候选角色或指定 Skill 没有实际执行')
        if not version_executed: violations.append('未观察到候选 Prompt 的真实会话版本分配')
        if result.get('error'): violations.append(result['error'])
        record = {'score': result['score'] if result.get('p0_pass') is True else 0.0, 'violations': violations,
            'feedback': {'checks': result.get('judged', {}), 'failures': result['tool_failures'],
                         'candidate_execution': result['candidate_execution']},
            'output': {'verdict': result['verdict'], 'metrics': result.get('metrics'),
                       'experience_reuse': result['experience_reuse'],
                       'tool_calls': len(calls), 'model_calls': result['model_calls'],
                       'tool_failures': len(result['tool_failures']),
                       'steps': [{'tool': row['name'], 'arguments': row['arguments'],
                                  'status': row['status'], 'failure': row['failure']} for row in calls],
                       'answer': result.get('transcript', '')},
            'execution_ref': str((execution / 'result.json').relative_to(self.output))}
        record['provider_failures'] = provider_failures
        record['task_score'] = record['score']
        # 完整任务通过优先；通过后比较每个真实执行步骤的完成效率。
        # 这是调用次数代理，不把模型与工具调用数冒充货币成本。
        record['optimization_score'] = (1.0 / (1 + result['model_calls'] + len(calls))
                                        if result['verdict'] == 'PASS' and not violations else 0.0)
        self.records.append({'id': case['id'], 'split': case.get('split'), **record})
        if interruption is not None:
            raise interruption
        return record

    def __call__(self, case, text, **kwargs):
        result = self.runner.run(self.run(case, text, **kwargs))
        if result['provider_failures']:
            raise TaskEvaluationUnavailable(result['execution_ref'])
        return result

    def reflect(self, messages):
        from app.infrastructure.context_usage import context_call_kind, context_usage_sink

        async def call():
            usage_token = context_usage_sink.set(self.reflection_usage.append)
            kind_token = context_call_kind.set('optimization_reflection')
            try:
                return (await self.model.ainvoke(messages if isinstance(messages,list) else
                    [{'role':'user','content':messages}])).text
            finally:
                context_call_kind.reset(kind_token)
                context_usage_sink.reset(usage_token)
        return self.runner.run(call())
