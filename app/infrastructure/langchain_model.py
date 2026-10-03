"""LangChain 官方模型扩展接口：SDK 传输，完整参数只解析一次。"""
import json
import time
from contextlib import AsyncExitStack
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import Field
from app.application.runtime.errors import ExecutionStopped
from app.infrastructure.budget import get_budget
from app.infrastructure.context_usage import record_context_usage, record_evaluation_evidence, context_request_details
from app.infrastructure.model_protocol import ResponseContract, ModelProtocolViolation
from app.infrastructure.operational_metrics import observe_model, observe_model_started
from app.infrastructure.prompt_cache import mark_messages, is_cache_rejection, read_cache_usage


class ChatModelAdapter(BaseChatModel):
    client: Any = Field(exclude=True)
    model_name: str
    streaming: bool = True
    max_tokens: int = 8192
    temperature: float | None = None
    reasoning_effort: str | None = None
    gateway: Any = Field(default=None, exclude=True)
    cache_mode: str = "passthrough"
    cache_policy: str = "static"
    cache_retry_enabled: bool = True

    @property
    def _llm_type(self):
        return "globex-chat-completions"

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("本项目模型入口使用 ainvoke，不输出半成品工具参数")

    async def aclose(self):
        await self.client.close()

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        declarations = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice == "any":
            tool_choice = "required"
        elif isinstance(tool_choice, str) and tool_choice not in {"auto", "none", "required"}:
            tool_choice = {"type":"function", "function":{"name":tool_choice}}
        return self.bind(tools=declarations, **({"tool_choice":tool_choice} if tool_choice else {}), **kwargs)

    def with_structured_output(self, schema=None, **kwargs):
        result = super().with_structured_output(schema, **kwargs)
        if kwargs.get("include_raw"):
            return result
        def required(value):
            if value is None:
                raise ValueError("模型未返回符合 schema 的结构化结果")
            return value
        return result | RunnableLambda(required)

    def prepare_request(self, messages, stop=None, **kwargs):
        wire = convert_to_openai_messages(messages)
        for message, item in zip(messages, wire, strict=True):
            if isinstance(message, AIMessage):
                if message.invalid_tool_calls:
                    item.setdefault("tool_calls", []).extend({"id":c["id"], "type":"function",
                        "function":{"name":c["name"], "arguments":c["args"]}} for c in message.invalid_tool_calls)
                if message.additional_kwargs.get("reasoning_content"):
                    item["reasoning_content"] = message.additional_kwargs["reasoning_content"]
        payload = {"model":self.model_name, "messages":wire, "stream":self.streaming,
                   "max_completion_tokens":self.max_tokens}
        if self.temperature is not None: payload["temperature"] = self.temperature
        if stop is not None: payload["stop"] = stop
        if self.reasoning_effort: payload["reasoning_effort"] = self.reasoning_effort
        payload.update({k:v for k,v in kwargs.items() if k in {
            "model", "tools", "tool_choice", "temperature", "max_completion_tokens", "response_format", "seed"}})
        if self.streaming: payload["stream_options"] = {"include_usage": True}
        if self.cache_mode == "explicit":
            payload["messages"], _ = mark_messages(payload["messages"], self.cache_policy)
        return payload

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        payload = self.prepare_request(messages, stop, **kwargs)
        try:
            return await self._request(payload, run_manager)
        except Exception as error:
            marked = any('cache_control' in b for m in payload['messages']
                         if isinstance(m.get('content'),list) for b in m['content'] if isinstance(b,dict))
            if not (self.cache_retry_enabled and marked and is_cache_rejection(error)):
                raise
            # 只重发被明确拒绝的缓存参数请求，不重放业务工具。
            for message in payload['messages']:
                if isinstance(message.get('content'),list):
                    for block in message['content']:
                        if isinstance(block,dict): block.pop('cache_control',None)
            return await self._request(payload, run_manager, cache_retry=True)

    async def _request(self, payload, run_manager, cache_retry=False):
        reservation = None
        budget = get_budget()
        if budget is not None:
            incoming = len(json.dumps([payload['messages'],payload.get('tools',[])],ensure_ascii=False).encode()) + 256
            maximum = min(payload['max_completion_tokens'], budget.remaining-incoming)
            reservation = budget.reserve(incoming+maximum) if maximum >= 64 else None
            if reservation is None: raise ExecutionStopped('budget_exhausted')
            payload = {**payload, 'max_completion_tokens':maximum}
        protocol = {'protocol_status':'unknown', 'response_model':None, 'cache_mode':self.cache_mode,
                    'cache_retry_without_markers':cache_retry,
                    'cache_marker_count':sum('cache_control' in b for m in payload['messages']
                        if isinstance(m.get('content'),list) for b in m['content'] if isinstance(b,dict))}
        contract = ResponseContract(payload,lambda **values:protocol.update(values))
        started = first = usage = None
        content, reasoning, calls, finish = [], [], {}, None
        refused = False
        try:
            async with AsyncExitStack() as stack:
                if self.gateway: await stack.enter_async_context(self.gateway.slot())
                with trace.get_tracer(__name__).start_as_current_span('globex.model', attributes={
                    'gen_ai.operation.name':'chat','gen_ai.request.model':payload['model']},
                    record_exception=False, set_status_on_exception=False) as span:
                    started=time.monotonic();observe_model_started()
                    details = context_request_details.get() or {}
                    for key in ('before_tokens', 'after_tokens', 'archived_result_count'):
                        if isinstance(details.get(key), int):
                            span.set_attribute('globex.context.' + key, details[key])
                    for key, value in details.get('request_parts_after', {}).items():
                        if isinstance(value, int):
                            span.set_attribute('globex.context.request.' + key, value)
                    recorded={**payload,'messages':[{k:v for k,v in m.items() if k!='reasoning_content'} for m in payload['messages']]}
                    record_evaluation_evidence('model_request',recorded)
                    try:
                        response=await self.client.chat.completions.create(**payload)
                        if payload['stream']:
                            async with response:
                                async for chunk in response:
                                    body=chunk.model_dump()
                                    usage=body.get('usage') or usage
                                    protocol['response_model']=body.get('model') or protocol['response_model']
                                    self._record_response(body,span)
                                    for choice in body.get('choices',[]):
                                        if choice['index'] != 0: raise ModelProtocolViolation('multiple_choices')
                                        finish=choice.get('finish_reason') or finish
                                        delta=choice.get('delta') or {}
                                        refused = refused or bool(delta.get('refusal'))
                                        if delta.get('reasoning_content'): reasoning.append(delta['reasoning_content'])
                                        if delta.get('content'):
                                            content.append(delta['content']);first=first or time.monotonic()
                                            if run_manager: await run_manager.on_llm_new_token(delta['content'])
                                        for raw in delta.get('tool_calls') or []:
                                            call=calls.setdefault(raw['index'],{'id':'','name':'','args':''})
                                            call['id']+=raw.get('id') or ''
                                            call['name']+=(raw.get('function') or {}).get('name') or ''
                                            call['args']+=(raw.get('function') or {}).get('arguments') or ''
                        else:
                            body=response.model_dump();self._record_response(body,span)
                            usage=body.get('usage');protocol['response_model']=body.get('model')
                            if len(body.get('choices',[])) != 1: raise ModelProtocolViolation('invalid_choices')
                            choice=body['choices'][0];finish=choice.get('finish_reason');message=choice['message']
                            content.append(message.get('content') or '')
                            refused = bool(message.get('refusal'))
                            if message.get('reasoning_content'): reasoning.append(message['reasoning_content'])
                            calls={i:{'id':c.get('id'),'name':c['function'].get('name'),'args':c['function'].get('arguments')}
                                   for i,c in enumerate(message.get('tool_calls') or [])}
                        if refused: finish = 'content_filter'
                        if finish in (None,'error','interrupted'):
                            raise ModelProtocolViolation('interrupted_response')
                        if calls and finish not in ({'tool_calls'} if payload['stream'] else {'tool_calls','stop'}):
                            raise ModelProtocolViolation('incomplete_tool_response')
                        ids=[c['id'] for c in calls.values()]
                        if any(not i for i in ids) or len(set(ids))!=len(ids):
                            raise ModelProtocolViolation('invalid_tool_identity')
                        contract.validate([c['name'] for c in calls.values()])
                        span.set_attribute('gen_ai.response.finish_reasons',[finish])
                        if finish in {'length','content_filter'}:
                            protocol['protocol_status'] = 'truncated' if finish=='length' else 'refused'
                        valid,invalid=[],[]
                        for call in calls.values():
                            try:
                                args=json.loads(call['args'])
                                if not isinstance(args,dict):raise ValueError('工具参数必须是 JSON 对象')
                                valid.append({'id':call['id'],'name':call['name'],'args':args})
                            except (ValueError,TypeError) as error:
                                invalid.append({**call,'error':str(error)})
                        if invalid:
                            protocol['protocol_status']='invalid_tool_arguments'
                            record_evaluation_evidence('model_argument_error',{'calls':invalid,'finish_reason':finish,
                                'span_id':format(span.get_span_context().span_id,'016x')})
                        incoming=(usage or {}).get('prompt_tokens');outgoing=(usage or {}).get('completion_tokens')
                        known=all(type(v) is int and v>=0 for v in (incoming,outgoing))
                        message=AIMessage(content=''.join(content),tool_calls=valid,invalid_tool_calls=invalid,
                            additional_kwargs={'reasoning_content':''.join(reasoning)} if reasoning else {},
                            usage_metadata={'input_tokens':incoming,'output_tokens':outgoing,'total_tokens':incoming+outgoing} if known else None,
                            response_metadata={'finish_reason':finish,'model_name':protocol['response_model']})
                        return ChatResult(generations=[ChatGeneration(message=message,generation_info={'finish_reason':finish})],
                                          llm_output={'model_name':protocol['response_model'],'token_usage':usage})
                    except BaseException as error:
                        if isinstance(error,ModelProtocolViolation):protocol['protocol_status']=error.code
                        span.set_attribute('error.type',type(error).__name__);span.set_status(Status(StatusCode.ERROR))
                        raise
                    finally:
                        incoming=(usage or {}).get('prompt_tokens');outgoing=(usage or {}).get('completion_tokens')
                        known=all(type(v) is int and v>=0 for v in (incoming,outgoing))
                        if reservation:reservation.settle(incoming+outgoing if known else None)
                        elapsed=(time.monotonic()-started)*1000
                        ttft=(first-started)*1000 if first is not None else None
                        protocol.update(read_cache_usage(usage or {}))
                        protocol['response_model_matches']=(protocol['response_model']==payload['model']) if protocol['response_model'] else None
                        if known:
                            span.set_attribute('gen_ai.usage.input_tokens',incoming)
                            span.set_attribute('gen_ai.usage.output_tokens',outgoing)
                        observe_model(input_tokens=incoming if known else None,output_tokens=outgoing if known else None,
                                      elapsed_ms=elapsed,ttft_ms=ttft,cost_usd=None)
                        # 观测故障不能覆盖正在传播的取消/传输异常。
                        import sys
                        propagating = sys.exc_info()[0] is not None
                        try:
                            record_context_usage(incoming if known else None,outgoing if known else None,elapsed,ttft_ms=ttft,cache=protocol)
                        except Exception:
                            if not propagating: raise
                        record_evaluation_evidence('model_response_contract',{'response_model':protocol['response_model'],
                                                                            'protocol_status':protocol['protocol_status'],
                                                                            'finish_reason':finish})
        finally:
            if reservation and not reservation.settled:reservation.settle(0 if started is None else None)

    @staticmethod
    def _record_response(body, span):
        record_evaluation_evidence('model_response_fragment',{'response_id':body.get('id'),'model':body.get('model'),
            'span_id':format(span.get_span_context().span_id,'016x'),
            'choices':[{k:({key:value for key,value in v.items() if key in {'content','tool_calls'}}
                          if k in {'delta','message'} and isinstance(v,dict) else v)
                       for k,v in choice.items() if k in {'index','finish_reason','delta','message'}}
                      for choice in body.get('choices',[])]})
