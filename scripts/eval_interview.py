"""隔离执行面试用例；复用 eval_regression 的执行、断言、HTTP 动作及报告。"""
import argparse
import asyncio
import json
import sys
import tempfile
from functools import partial
from datetime import datetime
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import yaml
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from app.infrastructure.settings import load_settings
from app.infrastructure.tracing import SanitizingSpanExporter
from scripts import eval_regression as runner
from scripts.eval.interview_runtime import LocalCollector, isolated_runtime, retrieval_evidence


async def replay_confirmations(client, container, result):
    """测试夹具明确要求的网络重试；对同一凭证重放，不创建第二张确认。"""
    requests = [e["payload"] for e in result["trace_events"] if e["type"] == "eval.http_action.result"]
    for saved in requests:
        before = await container.trade_store.get_inventory()
        buyer = saved["request"]["buyer_id"]
        count_before = (await container.trade_store.list_orders(buyer_id=buyer))["total"]
        response = await client.post(f"{runner.BASE_URL}/commerce/confirmations/{saved['confirmation_id']}/resolve", json=saved["request"])
        response.raise_for_status()
        after = await container.trade_store.get_inventory()
        count_after = (await container.trade_store.list_orders(buyer_id=buyer))["total"]
        same = response.json() == saved["response"]
        unchanged = before == after and count_before == count_after
        sku_ids = [item["sku_id"] for item in saved["expected_payload"]["items"]]
        result["trace_events"].append({"type":"eval.confirmation.replay", "payload":{
            "confirmation_id":saved["confirmation_id"], "same_response":same, "inventory_unchanged":unchanged,
            "inventory_before":{s:before[s] for s in sku_ids},"inventory_after":{s:after[s] for s in sku_ids},
            "orders_before":count_before,"orders_after":count_after}})
        if not same or not unchanged:
            raise ValueError("重复确认改变了交易结果或库存")
    if not requests:
        raise ValueError("没有可重放的真实 HTTP 确认")


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only")
    parser.add_argument("--cases", type=Path, default=runner.PROJECT_ROOT / "eval/interview.yaml")
    parser.add_argument("--replay", type=Path, help="仅重评分已有 case JSON，不调用模型或业务 API")
    parser.add_argument("--output", type=Path, default=Path("eval/verification") / ("interview-"+datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = parser.parse_args(argv)
    cases = yaml.safe_load(args.cases.read_text())["cases"]
    cases = [c for c in cases if not args.only or c["id"] == args.only]
    if not cases: parser.error("没有对应的用例")
    for case in cases:
        if any(runner.build_judge_rubric(case["rubric"], case["deterministic"]).values()):
            parser.error("面试用例必须采用确定性断言，主观项留人工审阅")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.replay:
        results=[]
        for case in cases:
            source=args.replay/(case["id"]+".json")
            result=json.loads(source.read_text())
            if result.get("error") is None:
                empty={"p0":[],"p1":[],"p2":[]}
                score,p0,verdict=runner.verify_fixed_trace_stability(empty,case["rubric"],case["deterministic"],result["trace_events"])
                result.update(score=score,p0_pass=p0,verdict=verdict,
                    judged=runner.apply_trace_evidence(empty,case["rubric"],case["deterministic"],result["trace_events"]))
            result.update(replay_source=str(source),execution_mode="replay_no_api")
            (args.output/(case["id"]+".json")).write_text(json.dumps(result,ensure_ascii=False,indent=2))
            results.append(result)
        (args.output/"report.md").write_text("# 已有轨迹离线重评分（未重新执行）\n\n"+runner.render_report(results))
        print(f"离线重评分：{sum(r['verdict']=='PASS' for r in results)}/{len(results)} PASS；{args.output}")
        return runner.exit_code_for_results(results)
    settings = load_settings()
    provider, exported = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(SanitizingSpanExporter(exported)))
    trace.set_tracer_provider(provider)
    runner.BASE_URL = "http://interview.test"
    results = []
    try:
        for case in cases:
            print(f"运行 {case['id']}", flush=True)
            exported.clear()
            started = perf_counter()
            case_events = []
            result = {"id":case["id"],"description":case["description"],"score":0,"p0_pass":False,
                "verdict":"ERROR","judged":{},"transcript":"","trace_events":[]}
            with tempfile.TemporaryDirectory(prefix="globex-interview-") as temp:
                try:
                    async with isolated_runtime(Path(temp)/"runtime", settings) as (app, container):
                        if case.get("skill"):
                            buyer = f"eval-buyer-{case['id']}-{runner._RUN_NAMESPACE}"
                            container.orchestrator._sessions._main_factory.buyer_skill_store.save(buyer, "通勤装备比较",
                                "研究通勤背包、耳机的使用场景和选购取舍时使用。",
                                "分别研究不同品类，先核对预算与目的地，再查询商品。候选只用核实过的SKU，给出证据引用；冲突参数不作比较依据。")
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=runner.BASE_URL) as client:
                            with patch.object(runner, "SessionEventCollector", partial(LocalCollector, events=case_events)):
                                async with asyncio.timeout(240):
                                    result = await runner.run_case(client, case, "", identity_policy=app.state.identity_policy)
                                    if case.get("replay_confirmation"):
                                        from scripts.eval.http_actions import BuyerAPIClient
                                        signed_client = BuyerAPIClient(client, runner.BASE_URL,
                                            app.state.identity_policy.issue(result["buyer_id"]))
                                        await replay_confirmations(signed_client, container, result)
                except Exception as error:
                    result.update(verdict="ERROR", score=0, p0_pass=False, error=f"{type(error).__name__}: {error}")
                    result["trace_events"] = case_events
                    if isinstance(error, runner.CaseExecutionError):
                        result.update(transcript=error.transcript, trace_events=error.trace_events)
            spans = [json.loads(span.to_json()) for span in exported.get_finished_spans()]
            model_spans = [s for s in spans if s["name"] == "globex.model"]
            def tokens(key):
                values = [s["attributes"].get("gen_ai.usage." + key) for s in model_spans]
                return sum(values) if values and all(type(v) is int for v in values) else None
            result["trace_metrics"] = {"model_calls":len(model_spans), "input_tokens":tokens("input_tokens"),
                "output_tokens":tokens("output_tokens"), "tool_calls":sum(s["name"] == "globex.tool" for s in spans),
                "scope":"实际 span，包括上下文整理模型；未知 Token 为 null"}
            result["trace_ids"] = sorted({span["context"]["trace_id"].removeprefix("0x") for span in spans})
            result["model"] = settings.llm_model
            result["execution_mode"] = "live_chat_model"
            result["retrieval"] = {**retrieval_evidence(result['trace_events']),
                "configured": {"embedding_model": settings.embedding_model, "embedding_version": settings.embedding_version,
                    "qdrant_collection": settings.qdrant_collection, "reranker_model": settings.reranker_model,
                    "hybrid_recall_enabled": settings.hybrid_recall_enabled}}
            result["manual_review"] = {"status":"PENDING", "questions":case["manual_review"]}
            result["wall_elapsed_ms"] = round((perf_counter()-started)*1000)
            result["scope"] = "真实聊天模型；生产API与主子图；临时SQLite与买家；商品及品类知识保留真实服务配置，实际商品检索执行见retrieval；共享索引只查询；不含支付"
            (args.output/(case["id"]+".json")).write_text(json.dumps(result,ensure_ascii=False,indent=2))
            (args.output/(case["id"]+".spans.json")).write_text(json.dumps(spans,ensure_ascii=False,indent=2))
            results.append(result)
            print(f"{case['id']}: {result['verdict']} ({result['wall_elapsed_ms']/1000:.1f}s)", flush=True)
            report = runner.render_report(results)
            report += "\n\n## 实际检索链路\n\n" + '\n'.join(
                f"- {r['id']}：语义查询 {r['retrieval']['semantic_searches']} 次；向量+rerank成功 {r['retrieval']['vector_rerank_searches']} 次；降级 {r['retrieval']['degraded_searches']} 次。"
                for r in results)
            report += "\n\n仅确定性检查结论；主观理由需人工审阅。耗时为单次样本，不能视作稳定性能指标。\n"
            (args.output/"report.md").write_text(report)
        print(f"证据目录：{args.output}", flush=True)
    finally:
        provider.shutdown()
    return runner.exit_code_for_results(results)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
