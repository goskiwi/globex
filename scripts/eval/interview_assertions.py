"""端到端用例的业务断言，复用现有事件，不解析模型 Markdown 金额。"""
import json


KINDS = {"answer_present", "confirmation_prepared", "presentation_matches", "presentation_terminal", "business_recovered", "delegation_delivered",
         "skill_loaded", "no_skill_load", "context_compacted", "no_execution_error", "vector_rerank_executed"}


def tool_results(events, name):
    return [e["payload"] for e in events if e.get("type") == "eval.tool_result" and e.get("payload", {}).get("tool") == name]


def check(assertion, events):
    kind = assertion["kind"]
    if kind == 'vector_rerank_executed':
        from scripts.eval.interview_runtime import retrieval_evidence
        data = retrieval_evidence(events)
        return data['external_pipeline_verified'], f"语义查询{data['semantic_searches']}次；向量+rerank成功{data['vector_rerank_searches']}次；降级{data['degraded_searches']}次（精确ID查询不要求rerank）"
    if kind == "answer_present":
        turns=sum(e.get("type") == "eval.turn.complete" for e in events)
        answers=[e.get("payload",{}).get("text") for e in events if e.get("type") == "final.result"]
        return turns > 0 and len(answers) == turns and all(isinstance(a,str) and a.strip() for a in answers) and all(e["payload"].get("status") == "completed" and e["payload"].get("stop_reason") is None for e in events if e.get("type") == "final.result"), f"完成轮次 {turns}，交付记录 {len(answers)}，非空回复 {sum(bool(isinstance(a,str) and a.strip()) for a in answers)}"
    if kind == "confirmation_prepared":
        from scripts.eval.http_actions import matches_expected
        rows = [r for e in events if e.get("type") == "eval.confirmations.snapshot"
                for r in e["payload"].get("confirmations", [])]
        valid = [r for r in rows if r.get("action") == "create" and r.get("status") == "pending"
                 and r.get("result") is None and matches_expected(r.get("payload"), assertion["expected_payload"])]
        return bool(valid), f"核验到匹配的待确认单 {len(valid)} 张"
    if kind == "no_execution_error":
        errors = [e for e in events if e.get("type") == "error" or (e.get("type") == "final.result" and (e["payload"].get("status") == "failed" or e["payload"].get("stop_reason")))]
        return not errors, f"运行错误事件 {len(errors)} 个"
    if kind in {"skill_loaded", "no_skill_load"}:
        loads = tool_results(events, "load_agent_skill_tool")
        return (any(r["state"] == "success" for r in loads) if kind == "skill_loaded" else not loads), f"Skill 调用 {len(loads)} 次"
    if kind == "business_recovered":
        failures = [i for i,e in enumerate(events) if e.get("type") == "eval.tool_result"
                    and e["payload"].get("tool") in assertion["failure_tools"] and e["payload"].get("state") == "error"]
        completed = [i for i,e in enumerate(events) if e.get("type") == assertion["success_event"] and e.get("payload",{}).get("hits")]
        return any(a < b for a in failures for b in completed), f"实际失败位置 {failures}；后续业务交付位置 {completed}"
    if kind == "context_compacted":
        rows = [e["payload"] for e in events if e.get("type") == "eval.context.operation"]
        return any(r.get("status") == "completed" for r in rows), f"整理状态 {[r.get('status') for r in rows]}"
    if kind == "delegation_delivered":
        rows = []
        for call in tool_results(events, "task_dispatch"):
            for value in call.get("result", []):
                if isinstance(value, str):
                    try: value = json.loads(value)
                    except ValueError: continue
                if isinstance(value, dict): rows.append(value)
        valid = [r for r in rows if r.get("status") in ("completed", "partial") and r.get("candidates") and r.get("evidence_refs")]
        product_categories = {}
        for call in tool_results(events, "product_search_tool"):
            for result in call.get("result", []):
                if isinstance(result, dict):
                    product_categories.update({hit['product_id']:hit.get('category')
                        for hit in result.get('hits', []) if hit.get('product_id')})
        covered = {product_categories.get(candidate.get('product_id'))
                   for row in valid for candidate in row['candidates']}
        required = set(assertion.get('categories', []))
        return (len(valid) >= assertion.get("minimum", 1) and required <= covered), f"有候选与证据的交接 {len(valid)} 次；状态 {[r.get('status') for r in rows]}；缺失品类 {sorted(required-covered)}"
    event_type = assertion["event"]
    deliveries = [(i, e["payload"]) for i, e in enumerate(events) if e.get("type") == event_type]
    if not deliveries:
        return False, f"未观察到 {event_type}"
    index, payload = deliveries[-1]
    if kind == "presentation_terminal":
        after = []
        for e in events[index+1:]:
            if e.get("type") == "eval.turn.complete": break
            after.append(e)
        observed_requests = any(e.get("type") == "eval.model_request" for e in events[:index])
        calls = sum(e.get("type") == "eval.model_request" for e in after)
        return observed_requests and calls == 0, f"观察到模型请求={observed_requests}；交付后额外模型调用 {calls} 次"
    if kind == "presentation_matches":
        from scripts.eval.http_actions import matches_expected
        actual = []
        for card in payload.get("hits", []):
            quote = card.get("landed_price") or {}
            actual.append({"sku_id":card.get("default_sku_id"), "quantity":card.get("quantity"),
                           "total_amount_minor":quote.get("total_amount_minor"), "currency":card.get("currency")})
        passed = matches_expected(sorted(actual,key=lambda r:r["sku_id"] or ""), sorted(assertion["items"],key=lambda r:r["sku_id"]))
        if "bundle_total_minor" in assertion:
            passed = passed and (payload.get("quote") or {}).get("total_amount_minor") == assertion["bundle_total_minor"]
        return passed, f"最终展示 {actual}；组合总额 {(payload.get('quote') or {}).get('total_amount_minor')}"
    raise ValueError(f"未知业务断言 {kind}")
