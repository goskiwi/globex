"""真实原生图、模型 HTTP 和工具执行的 OTel 层级，不导出业务正文。"""
import json

import httpx
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.application.agents.orchestrator import SubmitIntentInput
from app.infrastructure.tracing import SanitizingSpanExporter
from tests.native_model_helpers import client_model, completion
from tests.test_langgraph_runtime import _container


async def test_native_agent_model_tool_trace_and_usage_are_connected(tmp_path, monkeypatch):
    exported = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(SanitizingSpanExporter(exported)))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    private_text = "测试买家私有正文不导出"

    def handler(request):
        payload = json.loads(request.content)
        raw = completion()
        if not any(message["role"] == "tool" for message in payload["messages"]):
            raw["choices"][0]["finish_reason"] = "tool_calls"
            raw["choices"][0]["message"] = {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "lookup", "type": "function", "function": {
                    "name": "product_search_tool",
                    "arguments": '{"product_id":"P1003","sku_id":"P1003-S1"}',
                }}],
            }
        return httpx.Response(200, json=raw)
    model = await client_model(tmp_path, handler)
    container = await _container(tmp_path / "runtime", monkeypatch, model)
    try:
        with provider.get_tracer(__name__).start_as_current_span("request") as root:
            root_id = root.get_span_context().span_id
            result = await container.orchestrator.handle_intent(
                SubmitIntentInput("trace", "buyer", "zh-CN", "CNY", private_text),
                
            )
        assert result.error is None
        spans = exported.get_finished_spans()
        agent = next(span for span in spans if span.name == "globex.agent")
        models = [span for span in spans if span.name == "globex.model"]
        tools = [span for span in spans if span.name == "globex.tool"]
        assert agent.parent.span_id == root_id
        assert len(models) == 2 and len(tools) == 1
        assert all(span.parent.span_id == agent.context.span_id for span in [*models, *tools])
        assert all(span.attributes["gen_ai.usage.input_tokens"] == 1500 for span in models)
        assert private_text not in str([(span.attributes, span.events) for span in spans])
    finally:
        await container.shutdown()
        await model.aclose()
        provider.shutdown()
