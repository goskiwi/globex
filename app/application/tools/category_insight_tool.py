# -*- coding: utf-8 -*-
"""category_insight_tool

品类洞察工具（RAG）：回答"这个品类当前热卖什么、看哪些属性、价格区间、有什么坑"
这类选购常识问题，与 product_search_tool（出具体商品清单）分工明确。

工具返回框架无关的字符串型 ``ToolResult``，由 LangChain 适配器生成参数 schema。
"""
import json
from pathlib import Path
from app.infrastructure.rag.knowledge_retrieval import KnowledgeScope

from app.application.runtime.results import ToolResult, ToolResultState

from app.application.runtime.request_scope import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.rag.category_knowledge import (
    has_answerable_knowledge,
    keyword_fallback_insights,
    policy_fact_status,
)


def _chunk_text(content) -> str:
    """Chunk.content 是 TextBlock / DataBlock 而非纯字符串，统一归一为可序列化文本。"""
    if isinstance(content, str):
        return content
    text = getattr(content, "text", None)
    if text is not None:
        return text
    if isinstance(content, dict):
        return content.get("text") or str(content)
    return str(content)


def build_category_insight_tool(
    knowledge_base,
    bus: TradeEventBus,
    fallback_knowledge_dir: Path | None = None,
):
    async def category_insight_tool(question: str, scope: KnowledgeScope, top_k: int = 3) -> ToolResult:
        """查询品类洞察知识库：热卖款型、关键属性判断口径、价格区间、避坑点、跨境通则。

        适用于"这个品类怎么挑""现在流行什么""多少钱算合理""有什么坑"这类选购常识问题；
        需要具体商品清单与价格时用 product_search_tool。

        Args:
            question (`str`):
                自然语言问题，建议带上品类词，如"旅行装备怎么挑材质"、"美国免税额度多少"。
            scope (`KnowledgeScope`):
                category用于选购口径，填写目录品类；policy用于跨境规则，通用规则categories可为空。
            top_k (`int`):
                返回知识片段数量，默认 3。
        """
        scope=KnowledgeScope.model_validate(scope)
        context=ShoppingContext.current()
        parameters=(context.effective_search or {}).get('parameters',{}) if context else {}
        if parameters.get('ship_to'):scope=scope.model_copy(update={'region':parameters['ship_to']})
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {"tool": "category_insight_tool", "args": {"question": question, "top_k": top_k}},
        )
        try:
            from app.infrastructure.rag.knowledge_retrieval import search_knowledge
            results = await search_knowledge(knowledge_base, question, top_k,scope=scope)
        except Exception as err:  # noqa: BLE001 —— 知识库不可用时如实降级，不编造洞察
            fallback = (
                keyword_fallback_insights(question, fallback_knowledge_dir, top_k,scope=scope)
                if fallback_knowledge_dir is not None
                else []
            )
            if fallback:
                for insight in fallback:
                    insight["policy_fact_status"] = policy_fact_status(insight["metadata"])
                return ToolResult(data={
                        "insights": fallback,
                        "retrieval_mode": "keyword_fallback",
                    }, state=ToolResultState.SUCCESS)
            raise

        if not has_answerable_knowledge(results):
            reason = "当前知识库没有足够相关且可验证的资料，不能据此作确定性回答"
            return ToolResult(data={"insights": [], "unanswerable": True, "reason": reason}, state=ToolResultState.SUCCESS)

        insights = []
        for item in results:
            metadata = {
                key: item.chunk.metadata[key]
                for key in ("source_reference", "source_type", "published_at", "effective_from", "effective_to", "region", "version", "topic", "applicability")
                if item.chunk.metadata and key in item.chunk.metadata
            }
            insights.append({
                "content": _chunk_text(item.chunk.content),
                "source": item.chunk.metadata.get("source", item.document_id)
                if item.chunk.metadata
                else item.document_id,
                "score": round(item.score, 4),
                "metadata": metadata,
                "policy_fact_status": policy_fact_status(metadata),
            })
        return ToolResult(data={"insights": insights}, state=ToolResultState.SUCCESS)

    return category_insight_tool
