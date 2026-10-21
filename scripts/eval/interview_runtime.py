"""面试用例的隔离运行夹具；复用生产 API、主子图和交易账本，不是第二套 Agent。"""
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.infrastructure.context_usage import evaluation_evidence_sink
from app.infrastructure.persistence.sql.repositories import bootstrap_schema


@contextmanager
def capture_evaluation(events):
    """隔离评测记录模型测量；业务事件仍统一消费公开持久 SSE。"""
    token=evaluation_evidence_sink.set(lambda item: events.append({
        'type':'eval.'+item['kind'],'call_kind':item.get('call_kind'),'payload':item['payload']}))
    try:
        yield
    finally:
        evaluation_evidence_sink.reset(token)


@asynccontextmanager
async def isolated_runtime(folder: Path, settings, *, model_factory=None,isolated_knowledge=False):
    """仅隔离交易与买家，商品检索始终保留项目真实配置。"""
    from app.composition import build_container
    from app.presentation.server import build_app
    configuration = replace(settings, data_dir=folder, database_url=f"sqlite+aiosqlite:///{folder / 'globex.db'}",
        redis_url="", queue_enabled=False, tavily_api_key="",
        otlp_endpoint="", otlp_traces_endpoint="", langfuse_base_url="", langfuse_public_key="", langfuse_secret_key="",
        llm_fallback_model="", llm_max_retries=0, harness_enabled=True,
        identity_mode="hmac")
    folder.mkdir(parents=True)
    async with AsyncExitStack() as resources:
        if model_factory is not None:
            for module in ("app.application.agents.main_agent", "app.application.agents.search_agent", "app.application.agents.trade_agent",
                           "app.application.runtime.middleware", "app.infrastructure.llm"):
                resources.enter_context(patch(module + ".create_chat_model", model_factory))
        if isolated_knowledge:
            from qdrant_client import AsyncQdrantClient
            from app.infrastructure.rag.vector_knowledge import VectorKnowledgeBase
            from app.infrastructure.rag.category_knowledge import bootstrap_category_knowledge
            knowledge=VectorKnowledgeBase(configuration,client=AsyncQdrantClient(':memory:'))
            await bootstrap_category_knowledge(knowledge)
            resources.enter_context(patch('app.composition.build_category_knowledge_base',return_value=knowledge))
        with patch("app.composition.load_settings", return_value=configuration):
            container = await build_container()
        async def startup():
            # 仅初始化临时账本与会话，不调用生产建索引/知识库流程，避免写入共享Qdrant。
            await bootstrap_schema(container.db_engine)
            await container.trade_store.initialize_inventory(await container.product_repo.list_all())
            container.product_repo.bind_inventory(container.trade_store.get_inventory)
            await container.context_service.startup()
            await container.runs.startup()
        resources.enter_context(patch.object(container, "startup", startup))
        async def supplied_container():
            return container
        with patch("app.presentation.server.build_container", supplied_container):
            app = build_app()
            async with app.router.lifespan_context(app):
                yield app, container


def retrieval_evidence(events):
    """读取真实工具执行回执；不从配置或模型文字推测rerank已经成功。"""
    searches = []
    from scripts.eval.trace import tool_observations
    for result in tool_observations(events):
        payload = result['data']
        if result['name'] == 'product_search_tool' and result['status'] == 'success' and isinstance(payload, dict) and 'recall_strategy' in payload:
            searches.append({
                'recall_strategy': payload['recall_strategy'],
                'rerank_applied': payload.get('rerank_applied') is True,
                'hit_count': len(payload.get('hits') or []),
                'query_conditions': payload.get('query_conditions'),
            })
    semantic = [s for s in searches if s['recall_strategy'] != 'exact_id_lookup']
    verified = [s for s in semantic if s['recall_strategy'] in {'embedding_rerank', 'hybrid_rerank'} and s['rerank_applied']]
    return {'searches': searches, 'semantic_searches': len(semantic),
            'vector_rerank_searches': len(verified), 'degraded_searches': len(semantic) - len(verified),
            'external_pipeline_verified': bool(semantic) and len(verified) == len(semantic)}
