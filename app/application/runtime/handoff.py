"""子任务交付生命周期：候选提交 → 整轮校验 → 结果或有限修正。

复用 create_agent 的工具节点及 before/after_model 条件路由，不维护第二套执行循环。
"""
from dataclasses import dataclass
from copy import deepcopy
import json
import math
from typing_extensions import NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from app.application.agents.handoff import ResearchSubmission, SubagentResult
from app.application.runtime.errors import ExecutionStopped, raise_if_tool_stopped
from app.application.runtime.results import message_data
from app.infrastructure.context_products import PRODUCT_EVIDENCE_KINDS


def issue(code, field, message):
    return {"code": code, "field": field, "message": message}


class TaskEvidence:
    """当前子调用实际产生的证据；输入中提到对象不等于获得了推荐依据。"""
    def __init__(self, session_id):
        self.session_id = session_id
        self.refs = set()
        self.facts = {}
        self.qualification = {}
        self.quotes = {}
        self.identifiers = set()
        self.trade = []
        self.successful_tools = set()
        self.historical_refs = set()
        self._lookups = set()
        self.delegated = {}
        self.requirements = []

    @property
    def products(self):
        # 校验和交付读取同一份事实索引，不维护一份会与快照分离的 SKU 集合。
        result = {}
        for product_id, sku_id in self.facts:
            result.setdefault(product_id, set())
            if sku_id is not None:
                result[product_id].add(sku_id)
        return result

    def _products(self, cards, result_ref=None, historical=False, observed_at=None):
        product_fields = {"product_id", "title", "brand", "category", "product_type", "origin_country",
            "description", "highlights", "material_tags", "weight_kg", "dimensions_cm", "package_dimensions_cm",
            "rating_summary", "rating_is_live", "ships_to", "updated_at", "image_url", "image_kind", "image_alt",
            "source_platform", "canonical_product_id", "data_provenance"}
        for card in cards:
            product_id = card["product_id"]
            common = {key:deepcopy(value) for key,value in card.items() if key in product_fields}
            # 商品级资料不承载任何默认 SKU 的金额或库存。
            for sku in [None, *card.get("skus", [])]:
                key = (product_id, sku["sku_id"] if sku else None)
                old = self.facts.get(key)
                if old and not old["historical"] and historical:
                    continue
                facts = {"result_ref":result_ref, "historical":historical,
                         "observed_at":observed_at, "product":common,
                         "sku":deepcopy(sku)}
                if sku:
                    facts['sku'].pop('landed_price', None)
                self.facts[key] = facts
                if sku:
                    issues = list(sku.get('constraint_issues', []))
                    stock, price = sku.get('stock'), sku.get('price_major')
                    if type(stock) is int and stock <= 0:
                        issues.append('out_of_stock')
                    if stock is not None and type(stock) is not int:
                        issues.append('invalid_stock')
                    valid_price = type(price) in (int, float) and math.isfinite(price) and price >= 0
                    if price is not None and not valid_price:
                        issues.append('invalid_price')
                    self.qualification[key] = {'historical': historical, 'issues': issues,
                        'verified': (not historical and not issues and type(sku.get('stock')) is int
                                     and sku['stock'] > 0 and valid_price
                                     and isinstance(sku.get('currency'), str))}
                self.identifiers.add(product_id)
                if sku:
                    self.identifiers.add(sku["sku_id"])
            self._quotes(card.get("landed_price") or {}, result_ref, historical, observed_at)
            for sku in card.get('skus', []):
                self._quotes(sku.get('landed_price') or {}, result_ref, historical, observed_at)

    def _quotes(self, quote, result_ref, historical, observed_at):
        for line in quote.get("items", []):
            key = (line["product_id"], line["sku_id"], line["quantity"], quote["ship_to"], quote["currency"])
            old = self.quotes.get(key)
            if old and not old["historical"] and historical:
                continue
            self.quotes[key] = {"result_ref": result_ref, "historical": historical,
                "observed_at": observed_at, "ship_to": quote["ship_to"], "currency": quote["currency"],
                "line": deepcopy(line)}

    def capture_lookup(self, messages):
        for message in messages:
            if (not isinstance(message, ToolMessage) or message.name != "conversation_fact_lookup"
                    or message.status == "error"):
                continue
            data = message_data(message)
            if not isinstance(data, dict) or data.get("source") != "session_evidence":
                continue
            for record in data.get("records", []):
                key = (message.tool_call_id, record["result_ref"])
                if key in self._lookups: continue
                self._lookups.add(key)
                self.refs.add(record["result_ref"])
                self.historical_refs.add(record["result_ref"])
                if record.get("kind") in PRODUCT_EVIDENCE_KINDS:
                    self._products(record.get("data", {}).get("hits", []), record["result_ref"], True,
                                   record.get("data", {}).get("observed_at"))
                if record.get("kind") == "quote":
                    self._quotes(record.get("data", {}), record["result_ref"], True,
                                 record.get("data", {}).get("observed_at"))
            self.successful_tools.add("conversation_fact_lookup")

    def capture(self, event):
        if event.shopping_session_id != self.session_id or event.type != "tool.result":
            return
        payload = event.payload
        if not isinstance(payload, dict):
            return
        name = payload.get("tool")
        if name == "task_dispatch":
            self.delegated[payload.get("tool_call_id") or str(len(self.delegated))] = {
                "agent": payload.get("agent"), "status": payload.get("status"),
                "stop_reason": payload.get("stop_reason"),
                "candidates": [c["product_id"] for c in payload.get("candidates", [])]}
            return
        if payload.get("error"):
            return
        if name == "conversation_fact_lookup":
            self.capture_lookup([ToolMessage(content="", name=name, tool_call_id=payload.get("tool_call_id","lookup"),
                id=payload.get("tool_call_id"), artifact={"data":payload})])
            return
        if name not in {"product_search_tool", "get_product_details", "category_insight_tool", "web_search_tool",
                        "query_order_tool", "create_order_tool", "cancel_order_tool", "quote_products"}:
            return
        self.successful_tools.add(name)
        if payload.get("result_ref"):
            self.refs.add(payload["result_ref"])
        self._products(payload.get("hits", []), payload.get("result_ref"), False, payload.get("observed_at"))
        if name == "quote_products":
            self._quotes(payload.get("quote", {}), payload.get("result_ref"), False, payload.get("observed_at"))
        confirmation = payload.get("confirmation") or {}
        order = payload if name == "query_order_tool" else payload.get("order") or {}
        for record in [confirmation.get("payload") or {}, order]:
            for line in record.get("items", record.get("lines", [])):
                self.identifiers.update(str(line[k]) for k in ("product_id", "sku_id") if line.get(k))
        if confirmation:
            self.trade.append({"confirmation_id": confirmation["confirmation_id"],
                "action": confirmation["action"], "status": confirmation["status"],
                "expired": confirmation.get("expired", False)})
        if order:
            self.trade.append({"order_id": order["order_id"], "status": order["status"]})

    def stopped_result(self, reason):
        """停止时交回证据与缺口；检索命中不能自动升级为已筛选候选。"""
        return self.delivered(ResearchSubmission()).model_copy(update={
            'status':'partial' if self.successful_tools else 'failed','issues':[reason]})

    def stopped_text(self, reason):
        result = self.stopped_result(reason)
        lines = [str(ExecutionStopped(reason)) + '；' + result.summary]
        if self.products:
            lines.append(f"已读取 {len(self.products)} 件商品，完整资料保留在下列证据中；尚未完成筛选，不能当作推荐。")
        elif "product_search_tool" in self.successful_tools:
            lines.append("已完成的检索没有返回候选，不代表商品不存在。")
        elif not self.successful_tools:
            lines.append("本轮尚无可核验的业务成果。")
        other_reads = self.successful_tools & {"category_insight_tool", "web_search_tool", "conversation_fact_lookup"}
        if other_reads:
            lines.append("已执行部分资料查询，尚未形成完整结论。")
        if self.refs:
            lines.append("已保留证据引用：" + "、".join(sorted(self.refs)) + "。")
        if self.historical_refs:
            lines.append("其中历史证据不能替代当前价格、库存或订单状态。")
        # 同一实体按最后一次已核验结果展示，不把早先待确认状态覆盖到后续状态。
        trades = {r.get("confirmation_id") or r.get("order_id"): r for r in self.trade}
        for item in trades.values():
            if item.get("confirmation_id"):
                pending = item["status"] == "pending" and not item.get("expired")
                lines.append(f"确认单 {item['confirmation_id']}：" +
                             ("待用户确认，尚未执行交易。" if pending else f"状态 {item['status']}，请核对页面记录。"))
            else:
                lines.append(f"订单 {item['order_id']}：已核验状态 {item['status']}。")
        for task in self.delegated.values():
            lines.append(f"子任务 {task['agent']}：{task['status']}" +
                         (f"（{task['stop_reason']}）" if task["stop_reason"] else "") + "。")
            if task["candidates"]:
                lines.append("该子任务明确提交的候选：" + "、".join(task["candidates"]) + "；尚未由 Main 交付最终推荐。")
        lines.append("未完成部分需要继续核对；涉及写入但未取得结果的操作，请先查询状态，不要直接重放。")
        return "\n".join(lines)

    def delivered(self, submission):
        """只关联本子任务实际读取的商品与引用；模型不能填写或覆盖事实。"""
        payload = submission.model_dump()
        payload["candidates"] = [c.model_dump() for c in submission.selected_candidates]
        for candidate in payload["candidates"]:
            key = (candidate["product_id"], candidate.get("sku_id"))
            candidate["facts"] = deepcopy(self.facts[key])
            candidate["facts"]["quotes"] = [deepcopy(value) for quote_key,value in self.quotes.items()
                if key[1] is not None and quote_key[:2] == key]
        from app.application.usecases.research_report import research_summary
        from app.application.agents.handoff import TradeSubmission
        if isinstance(submission,TradeSubmission):
            status=submission.status
            payload['unmet_requirements']=payload.pop('unmet_constraints')
        else:
            constrained=any(v['issues'] for key,v in self.qualification.items()
                if key in {(c.product_id,c.sku_id) for c in submission.candidates})
            status='partial' if submission.unmet_requirements or constrained or not submission.candidates else 'completed'
            payload['status']=status
        summary=({'completed':'交易任务完成','partial':'交易任务部分完成','needs_input':'交易任务需要补充信息','failed':'交易任务未完成'}[status]
            if isinstance(submission,TradeSubmission) else research_summary(status,payload['candidates']))
        return SubagentResult(**payload, summary=summary,
            evidence_refs=sorted(self.refs),
            observed_product_count=len(self.products),
            observed_sku_count=sum(len(skus) for skus in self.products.values()))

    def validate(self, result):
        errors = []
        # 校验所选对象与规格，事实正文来自已读字段，不由模型重新填写。
        for index, candidate in enumerate(result.selected_candidates):
            if candidate.product_id not in self.products:
                errors.append(issue("candidate_unverified", f"candidates.{index}.product_id",
                                    "该候选没有实际读取证据，请移除，或将缺口交回主 Agent。"))
            elif candidate.sku_id is not None and candidate.sku_id not in self.products[candidate.product_id]:
                errors.append(issue("sku_mismatch", f"candidates.{index}.sku_id",
                                    "该 SKU 不属于候选商品的已读证据，请按工具结果修正。"))
        if isinstance(result,ResearchSubmission):
            for value in result.unmet_requirements:
                if value not in self.requirements:
                    errors.append(issue('requirement_unbound','unmet_requirements','只能引用本任务已有明确要求；其他局限写caveats'))
        if not self.successful_tools and (isinstance(result,ResearchSubmission) or result.status=='completed'):
            errors.append(issue('completion_without_evidence','submission','没有实际业务证据，不能提交研究或交易成果'))
        return errors


@dataclass
class HandoffContext:
    evidence: TaskEvidence


class HandoffState(AgentState):
    handoff_task: NotRequired[dict]
    handoff_submission: NotRequired[list[dict]]
    handoff_result: NotRequired[dict | None]
    handoff_feedback: NotRequired[list[dict]]
    handoff_attempts: NotRequired[int]


class HandoffMiddleware(AgentMiddleware[HandoffState, HandoffContext]):
    state_schema = HandoffState
    max_corrections = 2

    def __init__(self, schema=ResearchSubmission, bus=None, *, output_guard_enabled=True):
        self.schema, self.bus = schema, bus
        self.output_guard_enabled = output_guard_enabled

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        context = request.runtime.context
        if isinstance(context, HandoffContext) and isinstance(result, ToolMessage):
            context.evidence.capture_lookup([result])
        return result

    def _reject(self, state, errors, replacements=None):
        attempts = state.get("handoff_attempts", 0) + 1
        update = {"handoff_submission": [], "handoff_feedback": errors,
                  "handoff_attempts": attempts, "messages": replacements or []}
        if attempts > self.max_corrections:
            update.update(handoff_result=SubagentResult(status="failed",
                summary="子任务提交未通过校验，已停止修正；已完成业务结果仍保留。",
                issues=[f"{e['code']} ({e['field']}): {e['message']}" for e in errors]).model_dump(),
                jump_to="end")
        else:
            update["messages"].append(HumanMessage(name="handoff_feedback", content=json.dumps({
                "errors": errors, "instruction": "仅依据已有结果修正 SubagentResult；不要重新执行业务工具。",
                "corrections_remaining": self.max_corrections - attempts + 1}, ensure_ascii=False)))
            update["jump_to"] = "model"
        return update

    @hook_config(can_jump_to=["model", "end"])
    async def aafter_model(self, state, runtime):
        last = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        submissions = [c for c in last.tool_calls if c["name"] == "SubagentResult"] if last else []
        rejected = []
        if state.get("handoff_attempts") and last:
            # 修正交付时原业务工具已执行过，不允许重放确认准备等操作。
            rejected = [ToolMessage(name=c["name"], tool_call_id=c["id"], status="error",
                content="提交修正阶段未执行业务工具；请使用已有结果修正提交。")
                for c in last.tool_calls if c["name"] != "SubagentResult"]
        if submissions:
            return {"handoff_submission": submissions, "messages": rejected}
        if not last or not last.tool_calls or rejected:
            return self._reject(state, [issue("submission_missing", "submission",
                "请调用 SubagentResult 交付，普通文本不作为最终成果。")], rejected)
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        raise_if_tool_stopped(state["messages"])
        submissions = state.get("handoff_submission") or []
        if not submissions:
            return None
        context = runtime.context
        if not isinstance(context, HandoffContext):
            raise RuntimeError("子任务缺少服务端证据上下文")
        evidence = context.evidence
        evidence.capture_lookup(state["messages"])
        if len(submissions) != 1:
            errors = [issue("multiple_submissions", "submission", "一次只提交一份最终结果。")]
            result = None
        else:
            try:
                receipt = next((m for m in reversed(state["messages"])
                    if isinstance(m, ToolMessage) and m.tool_call_id == submissions[0]["id"]), None)
                if receipt is None or receipt.status == "error":
                    result = None
                    errors = [issue("submission_delivery_failed", "submission", receipt.text if receipt is not None else "提交工具未成功接收，请修正提交。")]
                else:
                    # 按 call_id 读取经现有工具中间件处理后的候选，避免绕过内容过滤。
                    payload = message_data(receipt)
                    result = self.schema.model_validate(payload["submission"])
                    errors = evidence.validate(result)
            except ValidationError as error:
                result = None
                errors = [issue("invalid_field", ".".join(map(str, e["loc"])) or "submission", e["msg"])
                          for e in error.errors(include_input=False, include_url=False, include_context=False)]
            except (ValueError, TypeError):
                result = None
                errors = [issue("invalid_submission_format", "submission", "提交工具未返回合法 JSON 对象，请重新提交。")]
        call_ids = {s["id"] for s in submissions}
        # 替换提交工具的接收回执，不依赖它在整批消息中的位置。
        replacements = [m.model_copy(update={"status": "error" if errors else "success",
            "content": json.dumps({"accepted": not errors, "errors": errors}, ensure_ascii=False)})
            for m in state["messages"] if isinstance(m, ToolMessage) and m.tool_call_id in call_ids]
        reason = state.get("execution_stop")
        accepted=evidence.delivered(result).model_dump() if not errors else None
        if accepted is not None and reason:
            accepted.update(status='partial',issues=[*accepted['issues'],reason])
        bus = self.bus
        for receipt in state["messages"]:
            if isinstance(receipt, ToolMessage) and receipt.tool_call_id in call_ids:
                execution_id = (receipt.artifact or {}).get("execution_id")
                if execution_id and bus is not None:
                    from app.infrastructure.security.output_guard import audit_output
                    payload = {
                        "tool_call_id": execution_id,
                        "decision": "rejected" if errors else "accepted",
                        "status": "failed" if errors else accepted["status"],
                        "result": accepted,
                        "errors": errors,
                    }
                    bus.publish(context.evidence.session_id, "handoff.result", payload,
                        public_payload={**payload, 'result':audit_output(accepted)[1], 'errors':audit_output(errors)[1]}
                            if self.output_guard_enabled else None)
        if errors:
            update = self._reject(state, errors, replacements)
            if update["jump_to"] == "model":
                # 此节点本身就在模型前，沿默认边进入模型，不能跳回自身。
                update.pop("jump_to")
            return update
        return {"handoff_result": accepted, "handoff_submission": [],
                "handoff_feedback": [], "messages": replacements, "jump_to": "end"}

    async def awrap_model_call(self, request, handler):
        if request.state.get("handoff_attempts"):
            return await handler(request.override(
                tools=[t for t in request.tools if t.name == "SubagentResult"], tool_choice="auto"))
        return await handler(request)
