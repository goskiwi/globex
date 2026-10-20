"""旧流程移除后的业务回归；使用真实工具和图，不调用外部模型。"""
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository


async def test_reference_or_excluded_id_does_not_turn_semantic_search_into_exact_lookup():
    usecase = CatalogSearchUseCase(InMemoryProductRepository())
    result = await usecase.execute(ProductSearchSpec(
        normalized_query='找类似 P1003 的其他通勤背包，排除 P1003',
        excluded_product_ids=('P1003',), top_k=10))
    assert result['recall_strategy'] != 'exact_id_lookup'
    assert result['hits']
    assert all(hit['product_id'] != 'P1003' for hit in result['hits'])
