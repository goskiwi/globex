# -*- coding: utf-8 -*-
"""LangChain/LangGraph 运行时韧性中间件。"""
from __future__ import annotations

import asyncio
import json
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from dataclasses import replace

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt

from app.infrastructure.context import ShoppingContext
from app.infrastructure.llm import create_chat_model
from app.infrastructure.resilience import DEFAULT_TIMEOUTS, _allow, _record_failure, _record_success
from app.infrastructure.transient import is_transient_error
from app.application.harness.assertions import check_schema
from app.infrastructure.security.content_filter import sanitize_tool_output
from app.infrastructure.context_usage import record_evaluation_evidence
from app.application.runtime.errors import ExecutionStopped
from app.infrastructure.operational_metrics import observe_tool_result


class BusinessToolMiddleware(AgentMiddleware):
    """一次工具执行的结果、状态更新与回执出口；不判断调用顺序。"""

    def __init__(self, loops, bus):
        self.loops, self.bus = loops, bus

    async def awrap_tool_call(self, request, handler):
        from app.application.runtime.results import message_data, project_data, tool_receipts, replace_receipts, receipt_failure
        name, call_id = request.tool_call["name"], request.tool_call["id"]
        session = ShoppingContext.current_session_id()
        execution = request.runtime.execution_info
        event_id = f"{execution.task_id}:{execution.node_attempt}:{call_id}"
        def emit(kind, delta="", state="success", data=None, failure=None):
            self.bus.publish(session, "tool.lifecycle", {"type":kind,"tool_call_id":event_id,
                "tool_call_name":name,"delta":delta,"state":state,"data":data,"failure":failure,
                "arguments":request.tool_call["args"] if kind=="TOOL_CALL_START" else None})
        emit("TOOL_CALL_START")
        emit("TOOL_CALL_DELTA", json.dumps(request.tool_call["args"],ensure_ascii=False))
        emit("TOOL_CALL_END");emit("TOOL_RESULT_START")

        def clean(value):
            if isinstance(value,str): return sanitize_tool_output(value)
            if isinstance(value,list):
                rows=[clean(v) for v in value]
                return any(hit for hit,_ in rows), [v for _,v in rows]
            if isinstance(value,dict):
                rows={k:clean(v) for k,v in value.items()}
                return any(hit for hit,_ in rows.values()), {k:v for k,(_,v) in rows.items()}
            return False,value

        with trace.get_tracer(__name__).start_as_current_span("globex.tool", attributes={
            "gen_ai.operation.name":"execute_tool","gen_ai.tool.name":name,"gen_ai.tool.call.id":event_id},
            record_exception=False,set_status_on_exception=False) as span:
            try:
                output = await handler(request)
                receipts = tool_receipts(output)
                if not receipts: raise TypeError("工具必须返回 ToolMessage 或含对应回执的 Command")
                processed=[]
                failed=any(r.status=="error" for r in receipts)
                for receipt in receipts:
                    notices=[]
                    data=message_data(receipt)
                    full=(receipt.artifact or {}).get("data",data)
                    public=(receipt.artifact or {}).get("event_data",full)
                    hit,data=clean(data); full_hit,full=clean(full)
                    _,public=clean(public)
                    if hit or full_hit:
                        notices.append("工具结果中的疑似提示词注入已过滤。")
                    if self.loops is not None:
                        loop=self.loops.observe(session,name,request.tool_call["args"],[data],receipt.status)
                        if loop:
                            notices.append(loop)
                            record_evaluation_evidence("tool_notice",{"tool":name,
                                "arguments":request.tool_call["args"],"reason":"same_arguments_and_result"})
                    if receipt.status == "success":
                        notices.extend("返回结构异常："+e["reason"] for e in check_schema(name,data).failures)
                    receipt=receipt.model_copy(update={"artifact":{**(receipt.artifact or {}),"data":full}})
                    receipt=project_data(receipt,data,notices=notices)
                    processed.append(receipt)
                    record_evaluation_evidence("tool_result",{"tool":name,"arguments":request.tool_call["args"],
                        "state":receipt.status,"result":[data],"stage":"normalized"})
                    payload={"tool":name,"tool_call_id":event_id}
                    if isinstance(public,dict):
                        payload.update(public)
                        if isinstance(public.get("hits"),list):payload["hit_count"]=len(public["hits"])
                    else:payload["result"]=public
                    if receipt.status=="error":
                        payload["error"]=receipt.text
                        payload['failure'] = receipt_failure(receipt)
                        span.set_status(Status(StatusCode.ERROR))
                    self.bus.publish(session,"tool.result",payload)
                    emit("TOOL_RESULT_TEXT_DELTA",receipt.text)
                    emit("TOOL_RESULT_END",state=receipt.status,data=public,failure=receipt_failure(receipt))
                observe_tool_result(failed=failed)
                return replace_receipts(output,processed)
            except BaseException as error:
                observe_tool_result(failed=True)
                span.set_attribute("error.type",type(error).__name__);span.set_status(Status(StatusCode.ERROR))
                emit("TOOL_RESULT_TEXT_DELTA","未取得完整回执；涉及写入请先核查状态。","error")
                emit("TOOL_RESULT_END",state="error",failure={'code':'interrupted' if isinstance(error,asyncio.CancelledError) else 'internal'})
                raise


class MemoryApprovalMiddleware(AgentMiddleware):
    """逐项确认同批记忆操作；所有决议收集完毕后才允许工具节点执行。

    interrupt 恢复会重新执行本节点，先前决议由 LangGraph 按调用顺序恢复。
    节点内没有数据库写入，拒绝以工具结果返回，不能被解释成执行成功。
    """

    def __init__(self, store=None):
        self.store = store

    async def aafter_model(self, state, runtime):
        from app.application.agents.tool_confirmation import MEMORY_WRITE_TOOLS
        messages = state["messages"]
        message = next((item for item in reversed(messages) if isinstance(item, AIMessage)), None)
        if message is None:
            return None
        names = {call["name"] for call in message.tool_calls}
        if names & MEMORY_WRITE_TOOLS and names - MEMORY_WRITE_TOOLS:
            return {"messages": [ToolMessage(
                content="记忆变更必须与业务工具分批调用；本批均未执行。变更完成后重新决策。",
                name=call["name"], tool_call_id=call["id"], status="error")
                for call in message.tool_calls]}
        rejected = []
        for call in message.tool_calls:
            if call["name"] not in MEMORY_WRITE_TOOLS:
                continue
            if (call["name"] in {"update_preference_tool", "forget_preference_tool"}
                    and getattr(self.store, "semantic_memory", False)):
                from app.application.memory.preference_selector import render_preference_lines
                context = ShoppingContext.current()
                if context is None:
                    raise ValueError("记忆审批缺少可信买家身份")
                facts = await self.store.list_by_buyer(context.buyer_id)
                args = call["args"]
                expected_text = args.get("previous_statement") if call["name"] == "update_preference_tool" else args.get("statement")
                if not any(fact.memory_id == args.get("memory_id")
                           and fact.version == args.get("expected_version")
                           and fact.statement == expected_text for fact in facts):
                    rejected.append(ToolMessage(content="目标记忆或版本已变化，未执行；请按最新 ID、版本与原文重新申请：\n"
                        + render_preference_lines(facts), tool_call_id=call["id"],
                        name=call["name"], status="error"))
                    continue
            response = interrupt({"action_id": call["id"], "action_requests": [{
                "name": call["name"], "args": call["args"],
                "description": "请确认或拒绝本次长期记忆变更",
            }], "review_configs": [{
                "action_name": call["name"], "allowed_decisions": ["approve", "reject"],
            }]})
            decisions = response.get("decisions") if isinstance(response, dict) else None
            if (not isinstance(decisions, list) or len(decisions) != 1
                    or decisions[0].get("type") not in {"approve", "reject"}):
                raise ValueError("记忆审批只接受单项批准或拒绝")
            if decisions[0]["type"] == "reject":
                rejected.append(ToolMessage(
                    content="买家已拒绝本次操作，未执行；未经新的明确请求不得重试。",
                    tool_call_id=call["id"], name=call["name"], status="error",
                ))
        return {"messages": rejected} if rejected else None


class GatewayModelMiddleware(AgentMiddleware):
    """持有网关并发名额直到一次模型调用完整结束，并提供备用模型回退。"""

    def __init__(self, settings, throttle, bus=None, *, client):
        self.settings, self.throttle, self.bus = settings, throttle, bus
        self.fallback = (
            create_chat_model(replace(settings, llm_model=settings.llm_fallback_model), client=client, stream=True, throttle=throttle)
            if settings.llm_fallback_model and settings.llm_fallback_model != settings.llm_model
            else None
        )

    async def awrap_model_call(self, request: ModelRequest, handler):
        last_error: Exception | None = None
        attempts = self.settings.llm_max_retries + 1
        for attempt in range(attempts):
            try:
                return await handler(request)
            except Exception as error:
                if not is_transient_error(error) or attempt + 1 >= attempts:
                    last_error = error
                    break
                await asyncio.sleep(min(6 * (3**attempt), 30))
        if self.fallback is not None and last_error is not None and is_transient_error(last_error):
            if self.bus is not None:
                self.bus.publish(ShoppingContext.current_session_id(), "model.fallback", {
                    "from": self.settings.llm_model, "to": self.settings.llm_fallback_model
                })
            return await handler(request.override(model=self.fallback))
        raise last_error or RuntimeError("模型调用失败")


class ToolResilienceMiddleware(AgentMiddleware):
    """在 LangGraph 工具节点边界执行超时和熔断。"""

    def __init__(self, registry, bus=None, timeouts=None):
        self.registry, self.bus = registry, bus
        self.timeouts = timeouts or DEFAULT_TIMEOUTS

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        name = request.tool_call["name"]
        call_id = request.tool_call["id"]
        if not await _allow(self.registry, name):
            detail = f"{name} 连续失败已熔断，暂不可用，请稍后再试"
            return ToolMessage(content="[error] " + detail, name=name, tool_call_id=call_id, status="error",
                               artifact={"error_code": "unavailable", "error_reason": "工具暂不可用：连续调用失败，已暂停访问", "executed": False})
        try:
            # task_dispatch 自己持有同一时限并交付部分结果，不能在外层先丢弃它的证据。
            result = await handler(request) if name == "task_dispatch" else await asyncio.wait_for(
                handler(request), self.timeouts.get(name, 30.0))
        except ExecutionStopped as error:
            return ToolMessage(content=str(error), name=name, tool_call_id=call_id, status="error",
                               artifact={"stop_reason": error.reason})
        except asyncio.TimeoutError:
            await _record_failure(self.registry, name)
            detail = f"{name} 执行超时"
            return ToolMessage(content="[error] " + detail, name=name, tool_call_id=call_id, status="error",
                               artifact={"error_code": "unavailable", "error_reason": "工具调用超时，执行结果尚未确认"})
        except Exception as error:
            transient = is_transient_error(error)
            code = "unavailable" if transient else "internal"
            # 未知程序故障也计入服务保护，但不因文案而自动重试。
            await _record_failure(self.registry, name)
            detail = f'{name} 服务暂不可用' if transient else f'{name} 执行异常：{type(error).__name__}；请勿原样重试'
            return ToolMessage(content='[error] ' + detail, name=name,
                               tool_call_id=call_id, status='error', artifact={"error_code": code,
                                   "error_reason": "外部服务连接或响应失败，执行结果尚未确认" if transient else "工具执行异常，执行结果尚未确认"})
        if isinstance(result, ToolMessage) and result.status == "error":
            if (result.artifact or {}).get("error_code") in {"unavailable", "internal"}:
                await _record_failure(self.registry, name)
        else:
            await _record_success(self.registry, name)
        return result


def runtime_middlewares(settings, throttle, circuit_registry, bus, loops=None) -> list[AgentMiddleware]:
    return [
        BusinessToolMiddleware(loops if settings.harness_enabled else None, bus),
        ToolResilienceMiddleware(circuit_registry, bus),
    ]
