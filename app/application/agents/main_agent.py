# -*- coding: utf-8 -*-
"""LangGraph 主代理装配与会话执行权管理。"""
from __future__ import annotations
from app.application.runtime.execution import ExecutionMiddleware, graph_step_limit

import json
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent

from app.application.agents.search_agent import SearchAgentFactory
from app.application.agents.trade_agent import TradeAgentFactory
from app.application.memory.preference_selector import PreferenceSelector
from app.application.prompts.loader import load_prompts
from app.application.runtime.tools import as_langchain_tool
from app.application.runtime.middleware import GatewayModelMiddleware, runtime_middlewares, MemoryApprovalMiddleware
from app.application.runtime.skills import SkillReferenceMiddleware
from app.application.runtime.context import RequestContextMiddleware
from app.application.runtime.working_state import WorkingStateMiddleware
from app.application.runtime.preferences import PreferenceStateMiddleware
from app.application.runtime.delivery import FinalDeliveryMiddleware
from app.application.tools.shopping_state_tool import build_shopping_state_tool
from app.application.tools.capability_tools import (
    CAPABILITY_POLICY,
    build_capability_tools,
)
from app.application.tools.forget_preference_tool import build_forget_preference_tool
from app.application.tools.remember_preference_tool import build_remember_preference_tool
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.application.tools.update_preference_tool import build_update_preference_tool
from app.domain.buyer.preference import PreferenceStore
from app.domain.session.ports.session_store import SessionStore
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.llm import create_chat_model
from app.infrastructure.settings import Settings
from app.infrastructure.tracing import record_prompt_assignment


@dataclass
class GraphSession:
    graph: Any
    config: dict
    tool_names: frozenset[str]
    context_policy: Any = None


class MainAgentFactory:
    def __init__(
        self,
        settings: Settings,
        search_factory: SearchAgentFactory,
        trade_factory: TradeAgentFactory,
        bus: TradeEventBus,
        preference_store: PreferenceStore,
        circuit_registry: Any,
        throttle: Any,
        *,
        checkpointer: Any,
        model_client: Any,

        loop_detector: Any = None,
        preference_selector: PreferenceSelector | None = None,
        capability_registry: Any = None,
        buyer_skill_store: Any = None,
        shopping_form_store: Any = None,
    ) -> None:
        self._model_client = model_client
        self._settings = settings
        self._search_factory = search_factory
        self._trade_factory = trade_factory
        self._bus = bus
        self._preference_store = preference_store
        self._throttle = throttle
        self._circuit_registry = circuit_registry
        self._checkpointer = checkpointer
        self._capability_registry = capability_registry
        self.capability_registry = capability_registry
        self.buyer_skill_store = buyer_skill_store
        self.shopping_form_store = shopping_form_store
        self.skill_catalog_mode = settings.skill_catalog_mode
        self._preference_selector = preference_selector or PreferenceSelector()
        self._loop_detector = loop_detector
        self._search_factory.bind_harness(loop_detector)
        self._trade_factory.bind_harness(loop_detector)
        for factory in (self._search_factory,self._trade_factory):
            factory.capability_registry = capability_registry
            factory.buyer_skill_store = buyer_skill_store

    def build(self) -> GraphSession:
        prompts = load_prompts()["main_agent"]
        tools = [*self._search_factory.build_tools(), *self._trade_factory.build_tools()]
        from app.application.tools.recommendation_tools import build_recommendation_tool, RecommendationInput, build_comparison_tool, ComparisonInput
        from app.application.tools.product_view_tool import build_product_view_tool, ProductViewInput
        tools.append(as_langchain_tool(build_product_view_tool(self._search_factory._catalog_search,
            self._search_factory.evidence_store, self._bus), args_schema=ProductViewInput))
        tools.append(as_langchain_tool(build_recommendation_tool(self._search_factory._catalog_search,
            self._search_factory.evidence_store, self._bus), args_schema=RecommendationInput))
        tools.append(as_langchain_tool(build_comparison_tool(self._search_factory._catalog_search,
            self._search_factory.evidence_store, self._bus), args_schema=ComparisonInput))
        tools.append(build_shopping_state_tool(self._search_factory.evidence_store))
        functions = [
            build_task_dispatch_tool(
                self._search_factory,
                self._trade_factory,
                self._bus,
                model_token_limit=self._settings.tool_result_limit,
            ),
            build_remember_preference_tool(self._preference_store, self._bus),
            build_update_preference_tool(self._preference_store, self._bus),
            build_forget_preference_tool(self._preference_store, self._bus),
        ]
        if self.shopping_form_store is not None:
            from app.application.tools.shopping_form_tool import build_shopping_form_tool
            from app.infrastructure.shopping_forms import ClarificationRequest
            tools.append(as_langchain_tool(build_shopping_form_tool(self.shopping_form_store,self._bus),
                                           args_schema=ClarificationRequest))
        tools.extend(as_langchain_tool(function) for function in functions)

        system_prompt = prompts["system_prompt"]
        if self.shopping_form_store is not None:
            system_prompt += (
                "\n缺少影响选择的关键条件或买家明确要求时调用 show_shopping_form。"
                "问题由你生成；表单不写长期记忆，也不能批准订单。调用后结束本轮。"
            )
        if self._capability_registry is not None:
            names = {tool.name for tool in tools}
            system_prompt += "\n\n" + CAPABILITY_POLICY
            tools.extend(
                as_langchain_tool(function)
                for function in build_capability_tools(
                    self._capability_registry,
                    names,
                    self._bus,
                    self.buyer_skill_store,
                )
            )

        skill_source = (SkillReferenceMiddleware(self._capability_registry,self.buyer_skill_store,
                                                {tool.name for tool in tools})
                        if self._capability_registry is not None else None)
        context_policy = RequestContextMiddleware(
            self._search_factory.evidence_store,
            create_chat_model(self._settings,client=self._model_client,stream=False,throttle=self._throttle),self._settings,
            system_prompt=system_prompt,tools=tools,skill_source=skill_source,
            working_state_mode=self._settings.context_state_mode)
        graph = create_agent(
            model=create_chat_model(self._settings,client=self._model_client, throttle=self._throttle, bus=self._bus),
            tools=tools,
            system_prompt=system_prompt,
            middleware=[
                *runtime_middlewares(
                    self._settings, self._throttle, self._circuit_registry, self._bus,
                    self._loop_detector,
                ),
                MemoryApprovalMiddleware(self._preference_store),
                GatewayModelMiddleware(self._settings, self._throttle, self._bus, client=self._model_client),
                ExecutionMiddleware(self._settings.agent_max_model_rounds, main=True,
                                    budget_lite_model=self._settings.llm_fallback_model,
                                    delivery_tools=("recommend_products", "compare_products", "show_product_details", "show_shopping_form")),
                FinalDeliveryMiddleware(),
                *([skill_source] if skill_source is not None else []),
                PreferenceStateMiddleware(self._preference_store, self._preference_selector,
                                          self._settings.preference_top_k),
                WorkingStateMiddleware(self._settings.context_state_mode),
                context_policy,
            ],
            checkpointer=self._checkpointer,
            name=prompts["name"],
        )
        return GraphSession(graph=graph, config={"recursion_limit": graph_step_limit(graph, self._settings.agent_max_model_rounds)}, tool_names=frozenset(tool.name for tool in tools),
                            context_policy=context_policy)


class SessionRegistry:
    """会话归属和 fencing 仍由业务存储控制；图状态由 LangGraph checkpointer 保存。"""

    def __init__(
        self,
        main_factory: MainAgentFactory,
        session_store: SessionStore,
        *,
        enforce_owner: bool = True,
        prompt_registry: Any = None,
    ) -> None:
        self._main_factory = main_factory
        self._session_store = session_store
        self._agents: dict[str, GraphSession] = {}
        self._claims: dict[str, Any] = {}
        self._enforce_owner = enforce_owner
        self._prompt_registry = prompt_registry

    async def get_or_create(self, shopping_session_id: str) -> GraphSession:
        from app.domain.session.ports.session_store import SessionOwnerMismatch
        from app.infrastructure.prompt_registry import PromptContractChanged

        context = ShoppingContext.current()
        if context is None or context.shopping_session_id != shopping_session_id:
            raise SessionOwnerMismatch("会话执行缺少可信的当前买家上下文")
        previous = self._claims.pop(shopping_session_id, None)
        claim = await self._session_store.claim(
            shopping_session_id,
            buyer_id=context.buyer_id,
            enforce_owner=self._enforce_owner,
        )
        if claim.state_json is not None:
            saved = json.loads(claim.state_json)
            if saved.get("runtime") != "langgraph" or saved.get("thread_id") != shopping_session_id:
                self._agents.pop(shopping_session_id, None)
                raise PromptContractChanged("旧运行时快照保留只读，请在新会话继续")
        ShoppingContext.set_session_fence(claim.fence)
        ShoppingContext.set_skill_catalog_mode(self._main_factory.skill_catalog_mode)
        capabilities = self._main_factory.capability_registry
        if capabilities is not None:
            import asyncio
            digest = await asyncio.to_thread(
                capabilities.bind_session,
                shopping_session_id,
                context.buyer_id,
                **({"allow_skill_updates": True} if self._main_factory.skill_catalog_mode == "append_only" else {}),
            )
            ShoppingContext.set_capability_digest(digest)
            record_prompt_assignment()
        if self._prompt_registry is not None:
            assignment = await self._prompt_registry.assign(shopping_session_id, context.buyer_id)
            ShoppingContext.set_prompt_assignment(assignment)
            record_prompt_assignment()
        if shopping_session_id not in self._agents or previous is None or previous.revision != claim.revision:
            session = self._main_factory.build()
            session.config = {**session.config, "configurable": {"thread_id": shopping_session_id}}
            self._agents[shopping_session_id] = session
        self._claims[shopping_session_id] = claim
        return self._agents[shopping_session_id]

    async def invalidate(self, shopping_session_id: str) -> None:
        self._agents.pop(shopping_session_id, None)
        self._claims.pop(shopping_session_id, None)

    async def persist(self, shopping_session_id: str) -> bool:
        claim = self._claims.get(shopping_session_id)
        if claim is None:
            return False
        payload = json.dumps(
            {"runtime": "langgraph", "thread_id": shopping_session_id, "schema": 1},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            saved = await self._session_store.save_claim(claim, payload)
            if self._claims.get(shopping_session_id) == claim:
                self._claims[shopping_session_id] = saved
            return True
        except BaseException:
            if self._claims.get(shopping_session_id) == claim:
                await self.invalidate(shopping_session_id)
            raise
