"""端到端用例的业务断言，复用现有事件，不解析模型 Markdown 金额。"""


from scripts.eval.trace import tool_observations, presentations, final_results, model_starts

KINDS = {"answer_present", "confirmation_prepared", "presentation_matches", "presentation_terminal", "business_recovered", "delegation_delivered",
         "skill_loaded", "no_skill_load", "context_compacted", "no_execution_error", "vector_rerank_executed"}


def tool_results(events, name):
    return [r for r in tool_observations(events) if r['name'] == name]


def deliveries(events, assertion):
    return presentations(events, assertion['presentation_kind'])


def candidate_identity(candidate):
    """原题要求明确SKU，必须与程序关联的商品及规格事实一致。"""
    product_id,sku_id=candidate.get('product_id'),candidate.get('sku_id')
    facts=candidate.get('facts') or {}
    if (isinstance(product_id,str) and product_id and isinstance(sku_id,str) and sku_id
            and (facts.get('product') or {}).get('product_id')==product_id
            and (facts.get('sku') or {}).get('sku_id')==sku_id):
        return product_id,sku_id
    return None


def check(assertion, events):
    kind = assertion["kind"]
    if kind == 'vector_rerank_executed':
        from scripts.eval.interview_runtime import retrieval_evidence
        data = retrieval_evidence(events)
        return data['external_pipeline_verified'], f"语义查询{data['semantic_searches']}次；向量+rerank成功{data['vector_rerank_searches']}次；降级{data['degraded_searches']}次（精确ID查询不要求rerank）"
    if kind == "answer_present":
        turns=sum(e.get("type") == "eval.turn.complete" for e in events)
        rows = final_results(events)
        valid = [r for r in rows if r['text'].strip() and r['status'] == 'completed' and r['stop_reason'] is None]
        return turns > 0 and len(rows) == len(valid) == turns, f"完成轮次 {turns}，有效交付 {len(valid)}"
    if kind == "confirmation_prepared":
        from scripts.eval.http_actions import matches_expected
        rows = [r for e in events if e.get("type") == "eval.confirmations.snapshot"
                for r in e["payload"].get("confirmations", [])]
        valid = [r for r in rows if r.get("action") == "create" and r.get("status") == "pending"
                 and r.get("result") is None and matches_expected(r.get("payload"), assertion["expected_payload"])]
        return bool(valid), f"核验到匹配的待确认单 {len(valid)} 张"
    if kind == "no_execution_error":
        errors = [r for r in final_results(events) if r['status'] == 'failed' or r['stop_reason']]
        return not errors, f"运行错误 {len(errors)} 个"
    if kind in {"skill_loaded", "no_skill_load"}:
        loads = tool_results(events, "load_agent_skill_tool")
        return (any(r["status"] == "success" for r in loads) if kind == "skill_loaded" else not loads), f"Skill 调用 {len(loads)} 次"
    if kind == "business_recovered":
        failures = [r['index'] for r in tool_observations(events)
                    if r['name'] in assertion['failure_tools'] and r['status'] == 'error']
        completed = [i for i,payload in deliveries(events,assertion) if payload.get("hits")]
        return any(a < b for a in failures for b in completed), f"实际失败位置 {failures}；后续业务交付位置 {completed}"
    if kind == "context_compacted":
        rows = [e["payload"] for e in events if e.get("type") == "eval.context.operation"]
        return any(r.get("status") == "completed" for r in rows), f"整理状态 {[r.get('status') for r in rows]}"
    if kind == "delegation_delivered":
        # 核对完整业务结果，不把供模型阅读的截断投影当成唯一证据。
        rows = [r['data'] for r in tool_results(events, 'task_dispatch') if r['status'] == 'success']
        sources = {r['scope']['task_id']:r for r in rows
                   if r.get('status') in ('completed','partial') and r.get('candidates') and r.get('evidence_refs')
                   and (r.get('scope') or {}).get('task_id')
                   and all(candidate_identity(c) for c in r['candidates'])}
        valid = {}
        for _,payload in deliveries(events,{'presentation_kind':'research'}):
            for report in payload.get('reports',[]):
                task_id=report.get('task_id')
                source=sources.get(task_id)
                if not source or task_id not in payload.get('task_ids',[]):continue
                identities={candidate_identity(c) for c in report.get('items',[])}
                if identities and None not in identities and identities <= {candidate_identity(c) for c in source['candidates']}:
                    valid[task_id]=report
        covered = {candidate['facts']['product'].get('category')
                   for report in valid.values() for candidate in report['items']}
        required = set(assertion.get('categories', []))
        return (len(valid) >= assertion.get("minimum", 1) and required <= covered), f"实际交付且SKU关联正确的研究 {len(valid)} 份；状态 {[r.get('status') for r in rows]}；缺失品类 {sorted(required-covered)}"
    accepted=deliveries(events,assertion)
    if not accepted:
        return False,f"未观察到已接纳的{assertion['presentation_kind']}"
    index,payload=accepted[-1]
    if kind == "presentation_terminal":
        after = []
        for e in events[index+1:]:
            if e.get("type") == "eval.turn.complete": break
            after.append(e)
        observed_requests = bool(model_starts(events[:index]))
        calls = len(model_starts(after))
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
