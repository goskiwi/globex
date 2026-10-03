"""硬过滤、粗排候选池、精排与输出分层；不以分批或盲目删文档掩盖超限。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.money import Money
from app.domain.catalog.product import Product
from app.domain.catalog.sku import Sku
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.catalog.ports.retrieval_ports import VectorHit
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository


def product(number, **fields):
    return Product(product_id=f'P{8000+number}', title=f'背包{number}', brand='测试',
        category=fields.pop('category', '旅行装备'), origin_country='CN', description=f'背包 通勤 型号{number}',
        ships_to=fields.pop('ships_to', ['CN']),
        skus=[Sku(f'P{8000+number}-S1', '标准', Money.from_major_units(fields.pop('price', 50), 'CNY'), fields.pop('stock', 5))],
        **fields)


def pipeline(products, reranker=None, **kwargs):
    index = SimpleNamespace(search=AsyncMock(side_effect=lambda emb, top_n: [
        VectorHit(p.product_id, 1/(i+1)) for i,p in enumerate(products[:top_n])]))
    usecase = CatalogSearchUseCase(InMemoryProductRepository(products),
        embedder=SimpleNamespace(embed=AsyncMock(return_value=[1.])), vector_index=index,
        reranker=reranker, **kwargs)
    return usecase, index


async def test_invalid_candidates_never_enter_rerank_and_reasons_survive():
    bad = [product(0, category='办公学习'), product(1, stock=0), product(2, ships_to=['US']), product(3, price=200)]
    good = [product(i) for i in range(4, 12)]
    reranker = SimpleNamespace(rerank=AsyncMock(side_effect=lambda query, docs: list(range(len(docs)))))
    usecase, _ = pipeline([*bad, *good], reranker, recall_candidates=8)
    result = await usecase.execute(ProductSearchSpec(normalized_query='背包', category='旅行装备',
        ship_to='CN', price_max_major=100, top_k=3))
    assert reranker.rerank.await_args.args[1] == [p.searchable_text() for p in good]
    assert result['hits'][0]['product_id'] == good[-1].product_id
    assert len(result['filtered_out']) == 3
    assert {row['reason'] for row in result['filtered_out']} == {'category_mismatch', 'out_of_stock', 'ship_to_unavailable'}
    assert result['retrieval_diagnostics']['eligible_candidates'] == 8
    assert result['retrieval_diagnostics']['rerank_candidates'] == 8


async def test_expanded_recall_is_filtered_then_coarse_limited_not_sent_whole():
    bad = [product(i, category='办公学习') for i in range(150)]
    good = [product(i) for i in range(150, 250)]
    async def rank(query, docs):
        assert len(docs) == 32
        assert docs == [p.searchable_text() for p in good[:32]]
        return list(range(len(docs)))
    usecase, index = pipeline([*bad, *good], SimpleNamespace(rerank=AsyncMock(side_effect=rank)))
    result = await usecase.execute(ProductSearchSpec(normalized_query='背包', category='旅行装备', top_k=12))
    assert [call.kwargs['top_n'] for call in index.search.await_args_list] == [32, 64, 128, 256]
    assert result['rerank_applied'] is True and len(result['hits']) == 12
    assert result['total_candidates'] == 100
    assert result['retrieval_diagnostics'] == {'recalled_candidates': 250, 'eligible_candidates': 100,
        'candidate_limit': 32, 'rerank_candidates': 32}


async def test_requested_output_can_exceed_default_pool_without_invalid_size():
    goods = [product(i) for i in range(60)]
    reranker = SimpleNamespace(rerank=AsyncMock(side_effect=lambda query, docs: [0.] * len(docs)))
    usecase, index = pipeline(goods, reranker)
    result = await usecase.execute(ProductSearchSpec(normalized_query='背包', top_k=50))
    assert index.search.await_args.kwargs['top_n'] == 50
    assert len(reranker.rerank.await_args.args[1]) == 50 and len(result['hits']) == 50


async def test_hybrid_union_is_preserved_but_only_coarse_pool_enters_rerank():
    goods = [product(i) for i in range(100)]
    index = SimpleNamespace(search_filtered=AsyncMock(return_value=[
        VectorHit(p.product_id, 1/(i+1)) for i,p in enumerate(goods[-32:])]))
    reranker = SimpleNamespace(rerank=AsyncMock(side_effect=lambda query, docs: [0.] * len(docs)))
    usecase = CatalogSearchUseCase(InMemoryProductRepository(goods),
        embedder=SimpleNamespace(embed=AsyncMock(return_value=[1.])), vector_index=index,
        reranker=reranker, hybrid_enabled=True, capture_retrieval_stages=True)
    result = await usecase.execute(ProductSearchSpec(normalized_query='背包', top_k=5))
    stages = result['retrieval_stages']
    assert len(stages['fused_candidates']) == 64
    assert set(stages['merged_candidates']) == set(stages['fused_candidates'])
    assert len(stages['rerank_candidates']) == 32
    assert len(reranker.rerank.await_args.args[1]) == 32
    assert set(stages['ranked_candidates']) == set(stages['rerank_candidates'])


async def test_all_rejected_skips_rerank_without_faking_execution():
    reranker = SimpleNamespace(rerank=AsyncMock(side_effect=AssertionError('无合格候选不精排')))
    usecase, _ = pipeline([product(i, stock=0) for i in range(10)], reranker)
    result = await usecase.execute(ProductSearchSpec(normalized_query='背包'))
    assert result['hits'] == [] and result['rerank_applied'] is False
    assert result['filtered_out'] and result['retrieval_diagnostics']['rerank_candidates'] == 0
    reranker.rerank.assert_not_awaited()


async def test_rerank_failure_keeps_filtered_coarse_pool_and_reports_no_success():
    goods = [product(i) for i in range(40)]
    usecase, _ = pipeline(goods, SimpleNamespace(rerank=AsyncMock(side_effect=RuntimeError('服务故障'))))
    result = await usecase.execute(ProductSearchSpec(normalized_query='背包'))
    assert result['recall_strategy'] == 'embedding_only' and result['rerank_applied'] is False
    assert [h['product_id'] for h in result['hits']] == [p.product_id for p in goods[:5]]
    assert result['retrieval_diagnostics']['rerank_candidates'] == 32


@pytest.mark.parametrize('value', [129, 256, True, 7])
def test_invalid_capacity_is_rejected_not_silently_clamped(value):
    with pytest.raises(ValueError, match='128'):
        CatalogSearchUseCase(InMemoryProductRepository([]), recall_candidates=value)
