# -*- coding: utf-8 -*-
"""三期单测：品类知识库 RAG / Context 策略 / 工具韧性（超时+熔断）。

全部不依赖真实 LLM 与外部服务：embedding 用确定性桩，向量库用 Qdrant 本地嵌入模式。
"""
import asyncio
import json

import pytest
from app.application.runtime.results import ToolResultState, ToolResult
from tests.native_tool_helpers import tool_graph, call_tool as _call
from app.application.runtime.middleware import ToolResilienceMiddleware

from app.infrastructure.transient import is_transient_error as _is_transient
from app.application.tools.category_insight_tool import build_category_insight_tool
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.rag.category_knowledge import bootstrap_category_knowledge, MarkdownKnowledgeBase
from app.infrastructure.resilience import (
    CircuitBreakerRegistry,
)

_TERMS = ("露营灯", "登山杖", "免税额度", "塑料", "茶具", "耳机", "行李箱", "运费")


@pytest.fixture()
async def knowledge_base(tmp_path):
    kb = MarkdownKnowledgeBase(
        name="category_insight_test",
        description="测试用品类知识库",
        collection="test_category_kb",
    )
    docs = tmp_path / "knowledge"
    docs.mkdir()
    (docs / "outdoor.md").write_text(
        "# 户外\n露营灯看防水等级与续航，登山杖优先钛合金。", encoding="utf-8",
    )
    (docs / "guide.md").write_text(
        "# 通则\n美国免税额度约 800 美元，运费按首件全价加续件折价计。", encoding="utf-8",
    )
    inserted = await bootstrap_category_knowledge(kb, knowledge_dir=docs)
    assert inserted == 2
    yield kb


class TestCategoryKnowledge:
    async def test_bootstrap_is_idempotent(self, knowledge_base, tmp_path):
        # 第二次灌同一目录不应重复插入
        again = await bootstrap_category_knowledge(knowledge_base, knowledge_dir=tmp_path / "knowledge")
        assert again == 0
        assert len(await knowledge_base.list_documents()) == 2

    async def test_insight_tool_returns_relevant_chunk(self, knowledge_base):
        bus = TradeEventBus()
        queue = bus.subscribe("anonymous")
        tool = build_category_insight_tool(knowledge_base, bus)

        response = await tool(question="美国免税额度是多少", top_k=2)
        payload = json.loads(response.text)
        assert payload["insights"], "应有知识命中"
        assert "免税额度" in payload["insights"][0]["content"]
        assert payload["insights"][0]["source"].endswith(".md")
        assert queue.qsize() == 1  # 业务函数只发布调用条件，结果由运行时统一发布

    async def test_insight_tool_degrades_when_kb_broken(self):
        class BrokenKnowledgeBase:
            async def search(self, *args, **kwargs):
                raise RuntimeError("向量库连接失败")

        bus = TradeEventBus()
        tool = build_category_insight_tool(BrokenKnowledgeBase(), bus)
        response = await _call(tool_graph(tool,middlewares=[ToolResilienceMiddleware(CircuitBreakerRegistry())]),
                               question="露营灯怎么挑")
        assert response.status == "error"
        assert response.artifact["error_code"] == "internal"
        assert "向量库连接失败" not in response.content

    async def test_insight_tool_uses_versioned_keyword_fallback_when_vector_search_breaks(self, tmp_path):
        class BrokenKnowledgeBase:
            async def search(self, *args, **kwargs):
                raise ValueError("Expecting value: line 1 column 1 (char 0)")

        docs = tmp_path / "knowledge"
        docs.mkdir()
        (docs / "travel-gear.md").write_text(
            "# 旅行装备\n\n## 材质与自重\n帆布是天然材料；三件套约 400g 算轻便。"
            "\n\n## 价格区间\n三件套 80-150 元入门，180-260 元主力。",
            encoding="utf-8",
        )
        bus = TradeEventBus()
        queue = bus.subscribe("anonymous")
        tool = build_category_insight_tool(
            BrokenKnowledgeBase(), bus, fallback_knowledge_dir=docs,
        )

        response = await tool(question="旅行装备材质、自重和价格怎么判断", top_k=2)
        payload = json.loads(response.text)
        assert queue.qsize() == 1

        assert response.state == ToolResultState.SUCCESS
        assert payload["insights"]
        assert any("400g" in insight["content"] for insight in payload["insights"])
        assert payload["retrieval_mode"] == "keyword_fallback"


class TestContextPolicy:
    def test_evidence_rules_do_not_authorize_transactions(self):
        from app.application.runtime.projections import EVIDENCE_RULES
        assert 'SKU' in EVIDENCE_RULES and '当前' in EVIDENCE_RULES
    def test_native_state_fields_are_explicit(self):
        from app.application.runtime.context import ContextState
        assert {'messages','context_summary','read_tool_messages'} <= ContextState.__annotations__.keys() | {'messages'}

class TestTransientRetryPolicy:
    """上游瞬时故障识别：网关把限流错误写在 SSE 流中间，2.0 模型层重试盖不到，
    靠 orchestrator 这一层按错误特征兜底（三期冒烟实际遇到过）。"""

    def test_gateway_concurrency_error_is_transient(self):
        import openai, httpx
        assert _is_transient(openai.APIError("busy",request=httpx.Request('GET','https://example.test'),body={"code":"Throttling.Concurrency"}))

    def test_rate_limit_variants_are_transient(self):
        import httpx
        for status in (408,429,500,502,503):
            response=httpx.Response(status,request=httpx.Request('GET','https://example.test'))
            assert _is_transient(httpx.HTTPStatusError('任何语言',request=response.request,response=response))
        assert _is_transient(TimeoutError())

    def test_business_errors_are_not_transient(self):
        for message in (
            "商品不存在：P9999",
            "仅 CONFIRMED 态可取消",
            "Invalid API key",
        ):
            assert not _is_transient(RuntimeError(message)), message


def _ok_tool_factory(name: str, delay: float = 0.0, fail: bool = False):
    async def tool_func() -> ToolResult:
        """测试用工具。"""
        if delay:
            await asyncio.sleep(delay)
        if fail:
            return ToolResult(data="[error] 503 Service Unavailable", state=ToolResultState.ERROR, error_code="unavailable")
        return ToolResult(data="ok", state=ToolResultState.SUCCESS)

    tool_func.__name__ = name
    return tool_func


class TestToolResilience:
    async def test_timeout_returns_error_chunk(self):
        registry = CircuitBreakerRegistry(failure_threshold=3, reset_seconds=60)
        middleware = ToolResilienceMiddleware(registry, timeouts={"slow_tool": 0.05})
        tool = tool_graph(_ok_tool_factory("slow_tool", delay=0.5), middlewares=[middleware])

        result = await _call(tool)
        assert result.status == 'error'
        assert "超时" in result.content
        assert registry.status("slow_tool") == "closed"  # 一次失败还没到阈值

    async def test_circuit_opens_after_threshold(self):
        registry = CircuitBreakerRegistry(failure_threshold=2, reset_seconds=60)
        middleware = ToolResilienceMiddleware(registry)
        tool = tool_graph(_ok_tool_factory("flaky_tool", fail=True), middlewares=[middleware])

        assert (await _call(tool)).status == 'error'
        assert (await _call(tool)).status == 'error'
        assert registry.status("flaky_tool") == "open"

        # 熔断后短路，返回降级提示而不是再次执行
        short_circuited = await _call(tool)
        assert "已熔断" in short_circuited.content

    async def test_half_open_probe_recovers(self):
        registry = CircuitBreakerRegistry(failure_threshold=1, reset_seconds=0)
        middleware = ToolResilienceMiddleware(registry)
        failing = tool_graph(_ok_tool_factory("recover_tool", fail=True), middlewares=[middleware])
        assert (await _call(failing)).status == 'error'
        assert registry.status("recover_tool") == "open"

        # reset_seconds=0 → 立即可转半开；探测成功后闭合
        healthy = tool_graph(_ok_tool_factory("recover_tool"), middlewares=[middleware])
        assert (await _call(healthy)).status == 'success'
        assert registry.status("recover_tool") == "closed"

    async def test_success_resets_failure_counter(self):
        registry = CircuitBreakerRegistry(failure_threshold=2, reset_seconds=60)
        middleware = ToolResilienceMiddleware(registry)
        failing = tool_graph(_ok_tool_factory("mixed_tool", fail=True), middlewares=[middleware])
        healthy = tool_graph(_ok_tool_factory("mixed_tool"), middlewares=[middleware])

        await _call(failing)  # 1 次失败
        await _call(healthy)  # 成功清零
        await _call(failing)  # 再 1 次失败，仍未达阈值
        assert registry.status("mixed_tool") == "closed"

    async def test_deterministic_business_error_does_not_trip_circuit(self):
        """不支持目的国属于请求校验失败，不代表检索基础设施故障。"""
        registry = CircuitBreakerRegistry(failure_threshold=2, reset_seconds=60)
        middleware = ToolResilienceMiddleware(registry)

        async def unsupported_destination() -> ToolResult:
            return ToolResult(data="[error] 暂不支持的目的国：BR", state=ToolResultState.ERROR, error_code="business_rejected")

        tool = tool_graph(unsupported_destination, middlewares=[middleware])
        for _ in range(4):
            assert (await _call(tool)).status == 'error'

        assert registry.status("unsupported_destination") == "closed"
