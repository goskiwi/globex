"""统一请求上下文：装配并计量 → 在剩余空间整理历史 → 校验最终请求。"""
import json
import re
from dataclasses import dataclass
from typing_extensions import NotRequired
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ExtendedModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from langchain_core.utils.function_calling import convert_to_openai_tool
from app.infrastructure.context import ShoppingContext
from app.infrastructure.context_products import token_estimate, result_identity
from app.application.runtime.errors import ContextCapacityError
from app.application.runtime.results import message_data, project_data
from app.infrastructure.budget import get_budget
from app.infrastructure.context_usage import context_call_kind, record_context_diagnostic, context_request_details
from opentelemetry import trace
from app.application.runtime.projections import share_identical_products, EVIDENCE_RULES, project_skill_history
from app.application.runtime.working_state import project_working_state
from app.application.runtime.tool_view import tool_view_budget
from app.application.runtime.context_summary import (
    ContextSummary, SUMMARY_TOKEN_LIMIT, source_entries, summary_instruction, build_summary, render_summary,
)
from app.infrastructure.context_statistics import ContextStatistics, RequestParts


class ContextState(AgentState):
    read_tool_messages: NotRequired[list[str]]
    context_summary: NotRequired[dict | None]
    context_statistics: NotRequired[dict]
    context_summary_failures: NotRequired[int]
    context_product_trigger: NotRequired[int]


@dataclass
class ComposedRequest:
    request: ModelRequest
    counts: dict


def request_parts(system, messages, tools, available):
    """按同一保守估算器逐部分计量，分项之和就是用于决策和最终校验的总量。"""
    sections={"system_tokens":token_estimate(system.model_dump(mode="json")),
              "tool_tokens":token_estimate([convert_to_openai_tool(tool) for tool in tools]),
              "skill_tokens":0,"state_tokens":0,"other_fixed_tokens":0,"history_tokens":0}
    for message in messages:
        key=("skill_tokens" if message.name in {"skill_reference","skill_catalog"} else
             "state_tokens" if message.name in {"shopping_state","shopping_state_delta"} else
             "other_fixed_tokens" if isinstance(message,SystemMessage) or message.name in RequestComposition.FIXED_NAMES else
             "history_tokens")
        sections[key]+=token_estimate(message.model_dump(mode="json"))
    # 原有整体 JSON 估算含数组分隔符；逐项计量保留这部分，不降低原容量保护。
    sections["protocol_tokens"]=token_estimate("["+(", "*len(messages))+"]")
    total=sum(sections.values())
    fixed=total-sections["history_tokens"]
    return RequestParts(**sections,fixed_tokens=fixed,total_tokens=total,input_limit=available,
                        history_budget=max(0,available-fixed)).model_dump()


class RequestComposition:
    """一轮请求的资料快照；所有候选历史都用相同规则重新装配和计量。"""
    FIXED_NAMES = {"skill_reference","skill_catalog","shopping_state","shopping_state_delta",
                   "memory_hint","trade_state","candidate_state","delegated_task"}

    def __init__(self, request, settings, skill_parts, working_mode):
        self.request, self.settings = request, settings
        self.history = list(request.messages)
        self.skill_parts, self.working_mode = skill_parts, working_mode
        self.available = settings.context_size - 8192 - max(4096,int(settings.context_size*.05))

    async def render(self, history):
        messages = project_working_state(project_skill_history(history),self.request.state,self.working_mode)
        messages.extend(self.skill_parts)
        request = self.request
        messages=share_identical_products(messages,
            compact_rules=getattr(self.settings,"context_compact_result_rules",False))
        messages=[m.model_copy(update={"artifact":None}) if isinstance(m,ToolMessage) else m for m in messages]
        system=SystemMessage(content=(request.system_message.text+"\n" if request.system_message else "")+EVIDENCE_RULES)
        counts=request_parts(system,messages,request.tools,self.available)
        if counts["fixed_tokens"] > self.available:
            raise ContextCapacityError("固定请求内容已超过安全容量，不能通过裁剪历史腾出空间")
        return ComposedRequest(request.override(messages=messages,system_message=system),counts)


def summary_boundary(messages, user_indexes, protected):
    """摘要只覆盖完整早期轮次；保留当前及两个完成轮次和全部未决调用。"""
    limit = user_indexes[-3] if len(user_indexes) >= 3 else 0
    limit = min(limit, min((i for i, m in enumerate(messages) if m.id in protected), default=len(messages)))
    calls = {call['id']: i for i, message in enumerate(messages) if isinstance(message, AIMessage)
             for call in [*message.tool_calls, *message.invalid_tool_calls]}
    results = {message.tool_call_id: i for i, message in enumerate(messages) if isinstance(message, ToolMessage)}
    for identifier in calls.keys() - results.keys():
        limit = min(limit, calls[identifier])
    cut = max((i for i in user_indexes if i <= limit), default=0)
    # 工具对即使跨越买家消息也必须位于同一侧，不能生成孤立工具结果。
    while True:
        crossing = [index for identifier, index in calls.items()
                    if index < cut <= results.get(identifier, -1)]
        if not crossing:
            return cut
        cut = max((i for i in user_indexes if i <= min(crossing)), default=0)


class RequestContextMiddleware(AgentMiddleware):
    state_schema = ContextState

    def __init__(self, store, model, settings, *, system_prompt, tools, skill_source=None,
                 working_state_mode=None, summary_enabled=True):
        self.store,self.model,self.settings=store,model,settings
        self.system_prompt,self.tools=system_prompt,list(tools)
        self.skill_source,self.working_state_mode=skill_source,working_state_mode
        self.summary_enabled=summary_enabled
        if settings.context_product_tokens<=0 or settings.context_target_tokens<=0:
            raise ValueError("上下文预算必须为正数")
        if not 0<getattr(settings,"context_prune_low_ratio",1)<=1:
            raise ValueError("无效整理低水位")

    async def prepare(self, request, *, force=False):
        if request.state.get('context_summary') is not None:
            ContextSummary.model_validate(request.state['context_summary'])
        parts=await self.skill_source.request_parts(request.state) if self.skill_source else []
        frame=RequestComposition(request,self.settings,parts,self.working_state_mode)
        history,updates=await self._compact(request.state,frame,force=force)
        final=await frame.render(history)
        if final.counts["total_tokens"] > frame.available:
            raise ContextCapacityError("整理后完整请求仍超出安全容量，原始记录保留")
        stats = ContextStatistics.model_validate(updates["context_statistics"]).model_dump()
        updates['context_statistics'] = stats
        record_context_diagnostic({"type": "request_compaction", **stats})
        span = trace.get_current_span()
        for key in ("before_tokens", "after_tokens", "archived_result_count"):
            span.set_attribute("globex.context." + key, stats.get(key, 0))
        for key, value in final.counts.items():
            span.set_attribute("globex.context.request." + key, value)
        if self.skill_source:
            await self.skill_source.validate_request(request.state)
        return final.request,updates,history

    async def compact_checkpoint(self, state, *, force=False):
        # 手动整理复用同一装配入口，构造时必须提供真实的系统提示与工具声明。
        request=ModelRequest(model=self.model,messages=state["messages"],system_message=SystemMessage(content=self.system_prompt),
                             tools=self.tools,state=state,runtime=None)
        _,updates,_=await self.prepare(request,force=force)
        return updates

    async def awrap_tool_call(self, request, handler):
        """同批工具共享上一请求的剩余容量；首次工具结果也计入包装预算。"""
        limit = self.settings.tool_result_limit
        statistics = request.state.get('context_statistics')
        if statistics is not None:
            stats = ContextStatistics.model_validate(statistics)
            last = next((m for m in reversed(request.state['messages']) if isinstance(m, AIMessage)), None)
            calls = len(last.tool_calls) if last is not None else 1
            reply_tokens = token_estimate(last.model_dump(mode='json')) if last is not None else 0
            remaining = stats.request_parts_after.input_limit - stats.after_tokens - reply_tokens
            # 最小引用仍要有空间；累积历史由后续统一装配整理，不通过丢弃未读事实腾位置。
            limit = min(limit, max(512, remaining // max(1, calls)))
        token = tool_view_budget.set(limit)
        try:
            return await handler(request)
        finally:
            tool_view_budget.reset(token)

    async def awrap_model_call(self, request, handler):
        prepared,updates,history=await self.prepare(request)
        token = context_request_details.set(updates['context_statistics'])
        try:
            response=await handler(prepared)
        finally:
            context_request_details.reset(token)
        # 模型成功后才提交历史整理；Command 在模型输出之后归并，不能抹掉本轮新回复/工具调用。
        if any(isinstance(m,RemoveMessage) and m.id==REMOVE_ALL_MESSAGES for m in updates["messages"]):
            updates={**updates,"messages":[*updates["messages"],*response.result]}
        updates={**updates,"read_tool_messages":[m.id for m in history if isinstance(m,ToolMessage)]}
        return ExtendedModelResponse(model_response=response,command=Command(update=updates))

    async def _compact(self, state, frame, *, force=False):
        context = ShoppingContext.current()
        if context is None:
            raise ValueError("上下文治理缺少可信会话")
        messages = list(frame.history)
        initial = await frame.render(messages)
        before = initial.counts["total_tokens"]
        available = frame.available
        target = min(self.settings.context_target_tokens, available)
        read = set(state.get("read_tool_messages", []))
        user_indexes = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
                        and m.name == context.buyer_id]
        latest_turn = user_indexes[-1] if user_indexes else 0
        latest_query = str(messages[latest_turn].content) if messages else ""
        selected = set(re.findall(r"P\d+(?:-S\d+)?", latest_query))
        from app.application.agents.shopping_state import protected_products
        selected.update(protected_products(state.get('shopping_work')))
        candidates = []
        for index, message in enumerate(messages):
            if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
                continue
            if message.name == "load_agent_skill_tool" and message.status == "success":
                continue  # 原文保留在 checkpoint/资料库，不为不会发送的正文触发商品裁剪。
            payload = message_data(message)
            if not isinstance(payload,dict):
                payload={'text':message.content,'tool':message.name} if len(message.content)>1000 else None
            if isinstance(payload, dict) and not payload.get('archived') and ('hits' in payload or len(message.content)>1000):
                candidates.append((index, message, payload))
        protected = {m.id for _, m, p in [c for c in candidates if 'hits' in c[2]][-2:]}
        protected.update(m.id for m in messages if isinstance(m, ToolMessage) and m.id not in read)
        newest={}
        for _,message,payload in reversed(candidates):
            for hit in payload.get('hits',[]):
                identity=result_identity(hit,payload.get('query_conditions',{}))
                if identity not in newest:
                    newest[identity]=message.id
                    if selected & set(re.findall(r'P\d+(?:-S\d+)?',json.dumps(hit))):protected.add(message.id)
        updates = []
        archived_count = 0
        total = sum(token_estimate(payload) for _, _, payload in candidates)
        initial_total=total
        ratio=getattr(self.settings,'context_prune_low_ratio',1)
        low_target=int(self.settings.context_product_tokens*ratio)
        trigger=self.settings.context_product_tokens if before>=target*.6 else state.get('context_product_trigger',self.settings.context_product_tokens)
        oversized={m.id for _,m,p in candidates if token_estimate(p)>getattr(self.settings,'tool_result_limit',20000)}
        eager=getattr(self.settings,'context_pruning_timing','pressure')=='after_use'
        current = initial
        pressure=force or eager or total>trigger or bool(oversized) or before > target
        for index, message, payload in candidates:
            if (message.id in protected or message.id not in read
                    or not pressure or (not force and not eager and total <= low_target and message.id not in oversized
                                        and current.counts["total_tokens"] <= target)):
                continue
            reference = payload.get("result_ref")
            evidence = await self.store.get(context.buyer_id, context.shopping_session_id, reference) if reference else None
            if evidence is None:
                reference = await self.store.save(context.buyer_id, context.shopping_session_id,
                    "products" if 'hits' in payload else 'tool_archive', payload)
            replacement = {
                "result_ref": reference, "historical": True, "archived": True,
                "query_conditions": payload.get("query_conditions", {}),
                "notice": "旧结果已读取并归档；使用 conversation_fact_lookup 回查，当前价格和库存需重新核验。",
            }
            if 'hits' in payload:
                replacement['identities'] = [{'product_id': hit['product_id'],
                    'sku_ids': [sku['sku_id'] for sku in hit.get('skus', [])]}
                    for hit in payload['hits']]
            changed = project_data(message, replacement)
            updates.append(changed)
            messages[index] = changed
            archived_count += 1
            total -= max(0, token_estimate(payload) - token_estimate(replacement))
            current = await frame.render(messages)
        current = await frame.render(messages)
        after = current.counts["total_tokens"]
        summary = state.get("context_summary")
        # 分界必须位于完整买家轮次开始处，不把工具调用与其结果拆开。
        cut = summary_boundary(messages, user_indexes, protected)
        budget = get_budget()
        failures=state.get('context_summary_failures',0)
        attempts=[]
        if (force or self.summary_enabled and after >= target) and self.model is not None and cut > 0 and (force or failures<3) and not (budget and budget.exhausted):
            head = messages[:cut]
            raw = [message.model_dump(mode="json") for message in head]
            # 归档仍保存完整原文，摘要只读取实际历史投影，避免把已失活流程总结成当前要求。
            projected_head = project_skill_history(head)
            entries = source_entries(projected_head)
            for item in projected_head:
                if not isinstance(item,ToolMessage):continue
                payload=message_data(item)
                if not isinstance(payload,dict):continue
                refs={payload['result_ref']} if payload.get('result_ref') else set()
                refs.update(row['result_ref'] for row in payload.get('records',[]) if row.get('result_ref'))
                for ref in refs:
                    evidence=await self.store.get(context.buyer_id,context.shopping_session_id,ref)
                    if evidence is None or evidence['kind']=='rejected_summary':raise ValueError('摘要来源证据缺失')
                    # 校验引用归属，但不把刚卸载的完整资料再次塞进摘要请求。
                    # 摘要仅复述当前投影和工作状态可见事实；缺失细节以后按引用回查。
            source = json.dumps({'sources': [entry.model_dump() for entry in entries]}, ensure_ascii=False)
            summary_rule = EVIDENCE_RULES + summary_instruction()
            summary_input = [SystemMessage(content=summary_rule), HumanMessage(content=source)]
            if request_parts(summary_input[0],summary_input[1:],[],available)["total_tokens"] + SUMMARY_TOKEN_LIMIT > available:
                raise ContextCapacityError("摘要来源超出模型安全窗口，原记录保留")
            reference = await self.store.save(context.buyer_id, context.shopping_session_id,
                                              "context_archive", {"messages": raw})
            kind = context_call_kind.set("summary")
            valid=False
            try:
                for attempt in range(2):
                    response = await self.model.ainvoke([
                        SystemMessage(content=summary_rule + ('上次选择无效，仅从已提供的source_id重新选择。' if attempt else '')),
                        HumanMessage(content=source),
                    ])
                    candidate=response.content
                    try:
                        summary = build_summary(candidate, entries, reference)
                        valid = True
                    except (ValueError, TypeError):
                        valid = False
                    attempts.append({'attempt':attempt+1,'status':'accepted' if valid else 'rejected'})
                    if valid:break
                    await self.store.save(context.buyer_id,context.shopping_session_id,'rejected_summary',
                        {'candidate':candidate,'source_ref':reference,'attempt':attempt+1})
                    if budget and budget.exhausted:break
            finally:
                context_call_kind.reset(kind)
            if valid:
                note = HumanMessage(content=render_summary(summary), name="context_summary", id="context-summary")
                removed = {m.id for m in head}
                updates = [m for m in updates if m.id not in removed]
                updates.extend(RemoveMessage(id=m.id) for m in head)
                updates.append(note)
                messages = [note, *messages[cut:]]
                updates = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]
                current = await frame.render(messages)
                after = current.counts["total_tokens"]
            else:
                if force:
                    raise ValueError("摘要未通过来源校验，原上下文保留")
                return list(frame.history), {'messages':[],'context_summary':state.get('context_summary'),
                    'context_summary_failures':failures+1,'context_statistics':{'status':'failed','summary_attempts':attempts,
                        'before_tokens':before,'after_tokens':before,'archived_result_count':0,
                        'request_parts_before':initial.counts,'request_parts_after':initial.counts}}
        if after > available:
            raise ContextCapacityError("受保护上下文超出模型容量，请缩小本轮范围")
        statistics = {"before_tokens": before, "after_tokens": after,
                      "status": "completed" if updates else "noop", 'summary_attempts':attempts,
                      "archived_result_count": archived_count,
                      "request_parts_before": initial.counts, "request_parts_after": current.counts}
        next_trigger=self.settings.context_product_tokens
        if ratio<1 and total>low_target and total<initial_total:
            batch_size=max((token_estimate(p) for _,m,p in candidates if m.id in protected),default=0)
            next_trigger=max(next_trigger,total+max(self.settings.context_product_tokens-low_target,min(self.settings.context_product_tokens,2*batch_size)))
        return messages, {"messages": updates, "context_summary": summary,'context_product_trigger':next_trigger,
                "context_statistics": statistics,'context_summary_failures':0 if attempts else failures}
