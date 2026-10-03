# -*- coding: utf-8 -*-
"""LangGraph 商品检索专家及其工具装配。"""
from __future__ import annotations
from app.application.runtime.execution import ExecutionMiddleware, graph_step_limit

from typing import Any

from langchain.agents import create_agent
from app.application.agents.handoff import HANDOFF_POLICY, build_submission_tool
from app.application.runtime.handoff import HandoffContext, HandoffMiddleware
from langchain_core.tools import BaseTool

from app.application.prompts.loader import load_prompts
from app.application.runtime.tools import as_langchain_tool
from app.application.runtime.middleware import runtime_middlewares, GatewayModelMiddleware
from app.application.runtime.skills import SkillReferenceMiddleware
from app.application.runtime.context import RequestContextMiddleware
from app.application.tools.capability_tools import build_capability_tools, CAPABILITY_POLICY
from app.application.tools.category_insight_tool import build_category_insight_tool
from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.tools.product_details_tool import build_product_details_tool
from app.application.tools.recommendation_tools import build_quote_tool
from app.application.tools.web_search_tool import build_web_search_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.llm import create_chat_model
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from app.infrastructure.rag.category_knowledge import KNOWLEDGE_DIR
from app.infrastructure.settings import Settings


class SearchAgentFactory:
    def __init__(
        self,
        settings: Settings,
        catalog_search: CatalogSearchUseCase,
        bus: TradeEventBus,
        knowledge_base: Any,
        circuit_registry: Any,
        throttle: Any,
        *, model_client: Any,
    ) -> None:
        self._model_client = model_client
        self._settings = settings
        self._catalog_search = catalog_search
        self._bus = bus
        self._knowledge_base = knowledge_base
        self._throttle = throttle
        self._circuit_registry = circuit_registry
        self.evidence_store = ContextEvidenceStore(settings.data_dir / "context_evidence.db")
        self.capability_registry = None
        self.buyer_skill_store = None

    def bind_harness(self, loop_detector: Any) -> None:
        self._loop_detector = loop_detector

    def build_tools(self) -> list[BaseTool]:
        functions = [
            build_product_details_tool(self._catalog_search, self._bus, self.evidence_store),
            build_quote_tool(self._catalog_search.pricing, self.evidence_store, self._bus),
            build_product_search_tool(
                self._catalog_search,
                self._bus,
                self.evidence_store,
            ),
            build_category_insight_tool(
                self._knowledge_base,
                self._bus,
                fallback_knowledge_dir=KNOWLEDGE_DIR,
            ),
            build_conversation_fact_lookup(
                self.evidence_store,
                mode=self._settings.context_lookup_mode,
            ),
        ]
        if self._settings.tavily_api_key:
            functions.append(build_web_search_tool(self._settings, self._bus))
        return [as_langchain_tool(function) for function in functions]

    def build(self):
        prompts = load_prompts()["sub_agents"]["search"]
        tools = self.build_tools()
        names = {t.name for t in tools}
        if self.capability_registry is not None:
            tools.extend(as_langchain_tool(fn) for fn in build_capability_tools(
                self.capability_registry,names,self._bus,self.buyer_skill_store))
        tools.append(build_submission_tool())
        system_prompt=prompts["system_prompt"]+"\n"+HANDOFF_POLICY+(
            CAPABILITY_POLICY if self.capability_registry is not None else "")
        skills=(SkillReferenceMiddleware(self.capability_registry,self.buyer_skill_store,names)
                if self.capability_registry is not None else None)
        context=RequestContextMiddleware(self.evidence_store,None,self._settings,system_prompt=system_prompt,
            tools=tools,skill_source=skills,summary_enabled=False)
        graph = create_agent(
            model=create_chat_model(self._settings, client=self._model_client, throttle=self._throttle, bus=self._bus),
            tools=tools,
            system_prompt=system_prompt,
            context_schema=HandoffContext,
            middleware=[
                *runtime_middlewares(
                self._settings, self._throttle, self._circuit_registry, self._bus,
                self._loop_detector,
            ),*([skills] if skills is not None else []),
                HandoffMiddleware(), GatewayModelMiddleware(self._settings, self._throttle, self._bus, client=self._model_client),
                ExecutionMiddleware(self._settings.agent_max_model_rounds, budget_lite_model=self._settings.llm_fallback_model,
                                    delivery_tools=("SubagentResult",)),
                context, ],
            name=prompts["name"],
        )
        return graph.with_config({"recursion_limit": graph_step_limit(graph, self._settings.agent_max_model_rounds)})
