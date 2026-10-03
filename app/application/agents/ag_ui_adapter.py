# -*- coding: utf-8 -*-
"""将 LangGraph 运行事件和业务结果投影为 AG-UI，不承担业务决策。"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable

from ag_ui.core import (
    AssistantMessage,
    BaseEvent,
    CustomEvent,
    MessagesSnapshotEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from app.infrastructure.eventbus import TradeEvent
from app.application.agents.execution_summary import TOOL_LABELS as _TOOL_LABELS, DELIVERY_TOOLS, label, request_summary, result_summary, delivery_summary

class AGUIRunAdapter:
    def __init__(self, request: RunAgentInput, emit: Callable[[BaseEvent], None], *, authoritative_state: dict | None = None) -> None:
        self.request = request
        self.emit = emit
        self._pending_delivery = None
        self.state: dict[str, Any] = {
            "recommendation": None,
            "comparison": None,
            "deliveredRunId": None,
            "productViews": [],
            "shoppingForms": [],
            "shoppingFilters": {},
            "confirmations": [],
            "toolApprovals": [],
            "skillUsages": [],
            "status": "queued",
            "process": {"runId": request.run_id,
                "userMessageId": next((m.id for m in reversed(request.messages) if m.role == 'user'), ''),
                "status": "queued", "steps": []},
        }
        if authoritative_state:
            self.state.update({k:copy.deepcopy(authoritative_state[k]) for k in ("recommendation","comparison","deliveredRunId","skillUsages","shoppingFilters") if k in authoritative_state})
        self.error: str | None = None
        self._text_open: set[str] = set()
        self._tool_open: set[str] = set()
        self._tool_names: dict[str, str] = {}
        self._tool_output: dict[str, list[str]] = {}
        self._skill_arguments: dict[str, dict] = {}
        self._tool_arguments: dict[str, dict] = {}
        self._delivery_steps: dict[str, str] = {}
        self._completed = False

    def _id(self, kind: str, original: str) -> str:
        return f"{self.request.run_id}:{kind}:{original}"

    def snapshot(self) -> None:
        # 队列消费者稍后才序列化，必须隔离后续原地状态修改。
        self.emit(StateSnapshotEvent(snapshot=copy.deepcopy(self.state)))

    def start(self) -> None:
        self.emit(RunStartedEvent(thread_id=self.request.thread_id, run_id=self.request.run_id))
        self.snapshot()

    def _progress(self, identifier: str, title: str, status: str, summary: str = '') -> None:
        entries = self.state["process"]["steps"]
        entry = next((entry for entry in entries if entry["id"] == identifier), None)
        if entry is None:
            entries.append({"id": identifier, "label": title, "status": status, "summary": summary})
        else:
            entry.update(label=title, status=status, summary=summary)

    def on_agent_event(self, event: Any) -> None:
        """保留框架原始的开始/增量/结束边界，不从工具名称推测调用关联。"""
        kind = getattr(event, "type", None)
        if kind == "REPLY_START":
            self.state["status"] = "running"
            self.state["process"]["status"] = "running"
            self.snapshot()
        elif kind == "TEXT_BLOCK_START":
            message_id = self._id("text", event.block_id)
            self._text_open.add(message_id)
            self.emit(TextMessageStartEvent(message_id=message_id, role="assistant"))
        elif kind == "TEXT_BLOCK_DELTA":
            if event.delta:
                self.emit(TextMessageContentEvent(
                    message_id=self._id("text", event.block_id), delta=event.delta,
                ))
        elif kind == "TEXT_BLOCK_END":
            message_id = self._id("text", event.block_id)
            self._text_open.discard(message_id)
            self.emit(TextMessageEndEvent(message_id=message_id))
        elif kind == "TOOL_CALL_START":
            call_id = self._id("tool", event.tool_call_id)
            self._tool_names[event.tool_call_id] = event.tool_call_name
            self._tool_open.add(call_id)
            self._tool_arguments[event.tool_call_id] = event.arguments if isinstance(event.arguments,dict) else {}
            if event.tool_call_name == "load_agent_skill_tool":
                self._skill_arguments[event.tool_call_id] = event.arguments or {}
            self.emit(ToolCallStartEvent(tool_call_id=call_id, tool_call_name=event.tool_call_name))
        elif kind == "TOOL_CALL_DELTA":
            if event.delta:
                self.emit(ToolCallArgsEvent(
                    tool_call_id=self._id("tool", event.tool_call_id), delta=event.delta,
                ))
        elif kind == "TOOL_CALL_END":
            call_id = self._id("tool", event.tool_call_id)
            self._tool_open.discard(call_id)
            # END 仅表示参数流结束；工具是否完成由 RESULT 表达。
            self.emit(ToolCallEndEvent(tool_call_id=call_id))
        elif kind == "TOOL_RESULT_START":
            name = event.tool_call_name
            self._tool_names[event.tool_call_id] = name
            self._tool_output[event.tool_call_id] = []
            if name == "load_agent_skill_tool":
                self._skill_reading(event.tool_call_id)
            arguments=self._tool_arguments.get(event.tool_call_id,{})
            self._progress(self._id("tool", event.tool_call_id), label(name,arguments), "running",request_summary(name,arguments))
            self.snapshot()
        elif kind == "TOOL_RESULT_TEXT_DELTA":
            self._tool_output.setdefault(event.tool_call_id, []).append(event.delta)
        elif kind == "TOOL_RESULT_DATA_DELTA":
            # 媒体结果保留其真实元数据，不在通用日志里重复传播大块 base64。
            self._tool_output.setdefault(event.tool_call_id, []).append(json.dumps({
                "media_type": event.media_type, "url": event.url,
            }, ensure_ascii=False))
        elif kind == "TOOL_RESULT_END":
            call_id = self._id("tool", event.tool_call_id)
            name = self._tool_names.get(event.tool_call_id, "工具")
            success = str(getattr(event.state, "value", event.state)).lower() == "success"
            content = "".join(self._tool_output.pop(event.tool_call_id, []))
            if name == "load_agent_skill_tool":
                self._skill_result(event.tool_call_id, event.data, success)
                success = success and next(item for item in self.state['skillUsages'] if item['toolCallId']==call_id)['status']=='used'
            self.emit(ToolCallResultEvent(
                message_id=self._id("result", event.tool_call_id),
                tool_call_id=call_id,
                content=content,
                role="tool",
            ))
            arguments = self._tool_arguments.pop(event.tool_call_id,{})
            if success and name=='update_shopping_state' and isinstance(event.data,dict) and isinstance(event.data.get('filters'),dict):
                self.state['shoppingFilters']=copy.deepcopy(event.data['filters'])
            step_status = 'completed' if success else 'failed'
            if success and name=='task_dispatch' and isinstance(event.data,dict):
                step_status = {'completed':'completed','partial':'partial','needs_input':'waiting_input','failed':'failed'}.get(str(event.data.get('status')),'unconfirmed')
            self._progress(call_id, label(name,arguments,event.data), step_status,
                result_summary(name,arguments,event.data,success,failure=event.failure))
            if success and name in DELIVERY_TOOLS:
                self._delivery_steps[DELIVERY_TOOLS[name]] = call_id
            self.snapshot()
        elif kind == "REPLY_END":
            reason = str(event.finished_reason).lower()
            if reason in {"error", "interrupted", "exceed_max_iters"}:
                self.error = "本轮执行未正常完成，请重试或缩小问题范围。"
        elif kind == "REQUIRE_USER_CONFIRM":
            self.state["status"]="awaiting_confirmation"
            self.state['process']['status']='waiting_confirmation'
            for call in event.tool_calls:
                item={"id":f"{event.reply_id}:{call.id}","tool":call.name,"label":_TOOL_LABELS.get(call.name,call.name),"arguments":call.input}
                if not any(p["id"]==item["id"] for p in self.state["toolApprovals"]):self.state["toolApprovals"].append(item)
                self._progress(self._id('approval',item['id']),item['label'],'waiting_confirmation','等待用户确认，尚未执行')
            self.snapshot()
        elif kind == "REQUIRE_EXTERNAL_EXECUTION":
            # 首版不接受 resume，遇到暂停必须如实结束，不能把未执行动作展示成成功。
            self.error = "本次操作需要人工确认或外部执行，当前流式入口尚未接入此续接流程。"

    def _skill_reading(self, original_id: str) -> None:
        call_id = self._id("tool", original_id)
        if any(item["toolCallId"] == call_id for item in self.state["skillUsages"]):
            return
        item = {"toolCallId": call_id, "status": "reading"}
        try:
            arguments = self._skill_arguments.get(original_id, {})
            for source, target in (("skill_id", "id"), ("version", "version")):
                value = arguments.get(source) if isinstance(arguments, dict) else None
                if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
                    item[target] = value
        except (ValueError, TypeError):
            pass
        self.state["skillUsages"].append(item)

    def _skill_result(self, original_id: str, loaded: Any, success: bool) -> None:
        self._skill_reading(original_id)
        call_id = self._id("tool", original_id)
        item = next(item for item in self.state["skillUsages"] if item["toolCallId"] == call_id)
        self._skill_arguments.pop(original_id, None)
        try:
            if (not success or not isinstance(loaded, dict) or loaded.get("kind") != "skill"
                    or loaded.get("authority") != "reference_only"
                    or not all(isinstance(loaded.get(key), str) and loaded[key] for key in ("id", "version", "title", "content_hash"))
                    or not re.fullmatch(r"[a-f0-9]{64}", loaded["content_hash"])
                    or any(item.get(key) is not None and item[key] != loaded[key] for key in ("id", "version"))):
                raise ValueError("不能确认方案读取成功")
            item.update(id=loaded["id"], version=loaded["version"], title=loaded["title"],
                        contentHash=loaded["content_hash"], status="used")
            item.pop("error", None)
        except (ValueError, TypeError):
            item.update(status="error", error="方案读取未成功或版本已变化，请刷新方案后重试。")

    def _close_skill_reads(self, message: str) -> None:
        for item in self.state["skillUsages"]:
            if item["status"] == "reading":
                item.update(status="error", error=message)

    def on_trade_event(self, event: TradeEvent) -> None:
        if event.shopping_session_id != self.request.thread_id:
            return
        payload = event.payload if isinstance(event.payload, dict) else {}
        if event.type == "tool.lifecycle":
            from app.application.runtime.events import ToolEvent
            self.on_agent_event(ToolEvent(**payload))
        elif event.type == "ui.surface":
            form = payload.get("form")
            if (isinstance(form, dict) and form.get("session_id") == self.request.thread_id
                    and isinstance(self.request.forwarded_props, dict)
                    and payload.get("buyer_id") == self.request.forwarded_props.get("buyerId")):
                self.state["shoppingForms"] = [copy.deepcopy(form)]
                for message in form.get("messages", []):
                    self.emit(CustomEvent(name="a2ui", value=copy.deepcopy(message)))
                self.snapshot()
        elif event.type in {"recommendation.result", "comparison.result", "product_view.result"}:
            # 工具准备的结果只有在本轮成功结束后，才成为页面交付。
            self._pending_delivery = (event.type.split(".")[0], copy.deepcopy(payload))
        elif event.type in {"confirmation.required", "confirmation.resolved"}:
            confirmation = payload.get("confirmation")
            if (isinstance(confirmation, dict) and isinstance(self.request.forwarded_props, dict)
                    and confirmation.get("buyer_id") == self.request.forwarded_props.get("buyerId")
                    and confirmation.get("session_id") == self.request.thread_id):
                previous = self.state["confirmations"]
                self.state["confirmations"] = [copy.deepcopy(confirmation), *[
                    item for item in previous if item["confirmation_id"] != confirmation["confirmation_id"]
                ]][:20]
                self.snapshot()
        elif event.type in {"agent.dispatch", "cache.hit", "model.fallback", "context.compressed", "error"}:
            self.emit(CustomEvent(name=event.type, value=payload))
            if event.type == "context.compressed":
                self.state["contextStatistics"] = payload
                self.snapshot()

    def _close_streams(self) -> None:
        # 中断/错误时关闭已打开的消息和参数流，客户端不会残留永久 loading。
        for message_id in sorted(self._text_open):
            self.emit(TextMessageEndEvent(message_id=message_id))
        self._text_open.clear()
        for call_id in sorted(self._tool_open):
            self.emit(ToolCallEndEvent(tool_call_id=call_id))
        self._tool_open.clear()
        self._tool_arguments.clear()

    def finish(self, final_text: str, status: str, stop_reason: str | None, *, product_delivery_complete: bool = False) -> None:
        if self._completed:
            return
        self._completed = True
        self._close_streams()
        self._close_skill_reads("未收到方案读取成功的结果。")
        # 用审核后的最终回答收口；保留客户端传入的全部历史与本轮用户消息。
        self.emit(MessagesSnapshotEvent(messages=[
            *self.request.messages,
            AssistantMessage(id=self._id("final", "answer"), content=final_text),
        ]))
        if status in {"completed", "partial"} and product_delivery_complete and not self.state["toolApprovals"] and self._pending_delivery is not None:
            kind, payload = self._pending_delivery
            if kind == "product_view":
                self.state["productViews"] = [{**payload, "runId": self.request.run_id}]
            else:
                self.state["recommendation"] = self.state["comparison"] = None
                self.state[kind] = payload
                self.state["deliveredRunId"] = self.request.run_id
            identifier = self._delivery_steps.get(kind)
            if identifier:
                name = next(name for name, value in DELIVERY_TOOLS.items() if value==kind)
                title = {'recommendation':'交付最终推荐','comparison':'交付比较结果','product_view':'展示商品详情'}[kind]
                self._progress(identifier,title,'completed',delivery_summary(name,payload,delivered=True))
        self._pending_delivery = None
        self.state["status"] = "awaiting_confirmation" if self.state["toolApprovals"] else status
        self.state["executionStatus"] = status
        self.state["stopReason"] = stop_reason
        self.state["productDeliveryComplete"] = product_delivery_complete
        self.state['process']['status'] = 'waiting_confirmation' if self.state['toolApprovals'] else {'failed':'failed','needs_input':'waiting_input'}.get(status,status)
        for entry in self.state['process']['steps']:
            if entry['status']=='running':
                entry.update(status='unconfirmed',summary='本轮已结束，未收到完整执行结果')
        self.snapshot()
        outcome={"type":"interrupt","interrupts":[{"id":p["id"],"reason":"tool_confirmation","message":p["label"]} for p in self.state["toolApprovals"]]} if self.state["toolApprovals"] else {"type":"success"}
        if status == "failed":
            self.emit(RunErrorEvent(message=final_text, code=stop_reason or "EXECUTION_FAILED"))
        else:
            self.emit(RunFinishedEvent(thread_id=self.request.thread_id, run_id=self.request.run_id,outcome=outcome))

    def fail(self, message: str, *, cancelled: bool = False, code: str | None = None) -> None:
        if self._completed:
            return
        self._completed = True
        self._close_streams()
        self._close_skill_reads("本轮已中断，方案读取尚未完成。")
        self.state["status"] = "cancelled" if cancelled else "error"
        self.state['process']['status']='cancelled' if cancelled else 'failed'
        for entry in self.state["process"]['steps']:
            if entry["status"] == "running":
                entry.update(status='cancelled' if cancelled else 'unconfirmed',summary='未收到完整执行结果')
        self.snapshot()
        self.emit(RunErrorEvent(message=message, code="CANCELLED" if cancelled else (code or "AGENT_ERROR")))
