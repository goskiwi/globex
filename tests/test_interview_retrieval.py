"""真实检索组件不得被评测关闭，执行降级不能被记为rerank通过。"""
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from scripts.eval.interview_runtime import isolated_runtime, retrieval_evidence
from scripts.eval.evidence import evaluate_trace_assertions
from tests.test_retrieval import _settings


def search(strategy, applied):
    return {'type': 'tool.result', 'payload': {'tool': 'product_search_tool',
        'recall_strategy': strategy, 'rerank_applied': applied, 'hits': [], 'query_conditions': {}}}


@pytest.mark.parametrize('events,expected', [
    ([search('embedding_rerank', True)], True),
    ([search('hybrid_rerank', True), search('exact_id_lookup', False)], True),
    ([search('embedding_only', False)], False),
    ([search('keyword_2gram', False)], False),
    ([search('bm25_rerank', True)], False),
    ([search('embedding_rerank', True), search('keyword_2gram', False)], False),
    ([search('exact_id_lookup', False)], False),
    ([], False),
])
def test_assertion_requires_actual_vector_and_rerank(events, expected):
    result = evaluate_trace_assertions([{'criterion': '真实检索', 'kind': 'vector_rerank_executed'}], events)
    assert result[0]['pass'] is expected
    assert retrieval_evidence(events)['external_pipeline_verified'] is expected


@pytest.mark.asyncio
async def test_live_keeps_native_retrieval_without_shared_index_writes(tmp_path, monkeypatch):
    import app.composition as composition
    from tests.test_langgraph_runtime import ScriptedModel
    settings = replace(_settings(tmp_path), llm_api_key='test-key', embedding_api_key='test-key',
        reranker_base_url='http://rerank.test/v1', reranker_model='rerank-test', hybrid_recall_enabled=True)
    bootstrap = AsyncMock()
    monkeypatch.setattr(composition, 'bootstrap_product_index', bootstrap)
    async with isolated_runtime(tmp_path/'live', settings, model_factory=lambda *a, **kw: ScriptedModel()) as (_, container):
        catalog = container.orchestrator._sessions._main_factory._search_factory._catalog_search
        assert catalog._embedder is container.embedder
        assert catalog._vector_index is container.vector_index
        assert catalog._reranker is not None
        assert container.orchestrator._sessions._main_factory._search_factory._knowledge_base is container.knowledge_base
        from app.infrastructure.semantic_memory import SemanticPreferenceStore, PreferenceDistiller
        factory = container.orchestrator._sessions._main_factory
        assert isinstance(factory._preference_store, SemanticPreferenceStore)
        assert isinstance(factory._preference_store.distiller, PreferenceDistiller)
        assert factory._preference_store is factory._preference_selector
        assert factory._preference_store.embedder is container.embedder
        assert factory._preference_store.path == tmp_path/'live'/'buyer_memory.db'
        assert container.settings.hybrid_recall_enabled is True
        assert container.settings.data_dir == tmp_path/'live'
        assert container.settings.redis_url == ''
        assert container.settings.semantic_cache_enabled is False
        assert not await container.vector_index._client.collection_exists(settings.qdrant_collection)
        bootstrap.assert_not_awaited()
