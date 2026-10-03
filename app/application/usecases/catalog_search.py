# -*- coding: utf-8 -*-
"""CatalogSearchUseCase

商品检索核心 UseCase，对齐参考实现五步流程：
    1. EmbeddingClient 把 normalized_query 向量化
    2. ProductVectorIndex.search(top_n) 拿候选 product_id（Qdrant，COSINE）
    3. ProductRepository.find_by_ids 还原 Product 聚合
    4. 硬条件过滤，按粗排相关性收敛精排候选池；不把扩大召回结果全量送入精排
    5. Reranker 对候选池精排，截取 top_k 并组装商品卡；失败时保留粗排顺序

降级链（recall_strategy 如实标注）：
    embedding_rerank → embedding_only → keyword_2gram（embedding 服务异常时兜底）

计价收敛设计：到手价在检索链路内联计算（TariffSchedule 规则内核），
不给 Agent 单独暴露比价/运费工具，减少不必要的工具调用轮次。

过滤可观测：被 ship_to / price_max_major 硬约束挡掉的候选以 filtered_out 摘要回传，
让模型能区分"库里没有这个商品"与"有但不满足约束"，不致于给出误导性结论。
"""
from __future__ import annotations

import logging
import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional

from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.catalog.ports.retrieval_ports import (
    EmbeddingClient,
    ProductVectorIndex,
    Reranker,
)
from app.domain.catalog.product import Product
from app.domain.catalog.sku import Sku
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.shipping.tariff_schedule import TariffSchedule
from app.application.usecases.pricing import PricingService
from app.application.usecases.product_media import product_media
from app.infrastructure.retrieval.bm25 import bm25_rank, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# 粗召回可扩展，但精排候选池不能超过现有BGE服务的单次文档容量。
MAX_RERANK_CANDIDATES = 128
_MAX_VECTOR_RECALL = 256

# 被硬约束挡掉的候选回传条数上限（只回摘要，避免上下文膨胀）
_FILTERED_OUT_LIMIT = 3


@dataclass(frozen=True)
class ProductCard:
    product_id: str
    title: str
    brand: str
    category: str
    origin_country: str
    price_major: float
    currency: str
    source_price_major: float
    source_currency: str
    highlights: list[str]
    skus: list[dict]
    score: float
    landed_price: Optional[dict]  # ship_to 命中时的到手价明细，未命中为 None
    source_platform: str
    canonical_product_id: str
    material_tags: list[str]
    weight_kg: float | None
    # 展示资料与具体默认 SKU 的价库存一起组装。
    description: str = ""
    rating_summary: dict[str, float | int] | None = None
    rating_is_live: bool = False  # 当前目录是评测快照，评分不来自实时平台查询。
    ships_to: list[str] = field(default_factory=list)
    dimensions_cm: dict[str, float] = field(default_factory=dict)
    package_dimensions_cm: dict[str, float] = field(default_factory=dict)
    updated_at: str = ""
    default_sku_id: str = ""
    image_url: str | None = None
    image_kind: str = "placeholder"
    image_alt: str = "暂无商品图片"
    source_language: str = ""
    source_locale: str = ""
    data_provenance: str = ""

    def to_dict(self) -> dict:
        card = {
            "product_id": self.product_id,
            "title": self.title,
            "brand": self.brand,
            "category": self.category,
            "origin_country": self.origin_country,
            "price_major": self.price_major,
            "currency": self.currency,
            "source_price_major": self.source_price_major,
            "source_currency": self.source_currency,
            "highlights": self.highlights,
            "skus": self.skus,
            "score": round(self.score, 4),
            "source_platform": self.source_platform,
            "canonical_product_id": self.canonical_product_id,
            "material_tags": self.material_tags,
            "description": self.description,
            "rating_summary": self.rating_summary,
            "rating_is_live": self.rating_is_live,
            "ships_to": self.ships_to,
            "dimensions_cm": self.dimensions_cm,
            "package_dimensions_cm": self.package_dimensions_cm,
            "updated_at": self.updated_at,
            "default_sku_id": self.default_sku_id,
            "image_url": self.image_url,
            "image_kind": self.image_kind,
            "image_alt": self.image_alt,
        }
        if self.landed_price is not None:
            card["landed_price"] = self.landed_price
        if self.weight_kg is not None:
            card["weight_kg"] = self.weight_kg
        if self.source_language:
            card.update(source_language=self.source_language, source_locale=self.source_locale)
        if self.data_provenance:
            card["data_provenance"] = self.data_provenance
        return card


def tokenize(text: str) -> set[str]:
    """极简分词：空格切词 + 中文连续段落的 2-gram（关键词降级召回用）。"""
    terms: set[str] = set()
    for chunk in text.lower().split():
        terms.add(chunk)
        # 对含 CJK 的 chunk 补 2-gram，缓解中文无空格问题
        if any("\u4e00" <= ch <= "\u9fff" for ch in chunk) and len(chunk) >= 2:
            terms.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return terms


class CatalogSearchUseCase:
    def __init__(
        self,
        product_repo: ProductRepository,
        embedder: Optional[EmbeddingClient] = None,
        vector_index: Optional[ProductVectorIndex] = None,
        reranker: Optional[Reranker] = None,
        tariff_schedule: Optional[TariffSchedule] = None,
        pricing: PricingService | None = None,
        hybrid_enabled: bool = False,
        hybrid_lexical_weight: float = 1.0,
        hybrid_vector_weight: float = 1.0,
        recall_candidates: int = 32,
        capture_retrieval_stages: bool = False,
    ) -> None:
        # 仅评测主动开启；记录混合检索的实际阶段，不改变排序或向模型增加字段。
        self._capture_retrieval_stages = capture_retrieval_stages
        self._hybrid_enabled = hybrid_enabled
        self._fusion_weights = (hybrid_lexical_weight, hybrid_vector_weight)
        reciprocal_rank_fusion([], [], weights=self._fusion_weights)
        if type(recall_candidates) is not int or not 8 <= recall_candidates <= MAX_RERANK_CANDIDATES:
            raise ValueError(f"recall_candidates 须为 8 到 {MAX_RERANK_CANDIDATES} 的整数，表示精排候选池容量")
        self._recall_candidates = recall_candidates
        self._product_repo = product_repo
        self._embedder = embedder
        self._vector_index = vector_index
        self._reranker = reranker
        self.pricing = pricing or PricingService(product_repo, tariff_schedule)
        self._tariff = self.pricing.tariff

    async def execute(self, spec: ProductSearchSpec) -> dict:
        # 稳定实体 ID 直接核对权威目录，不能用向量 top-N 判断商品是否存在。
        # 结构化标识优先；旧查询文字中的 ID 仍兼容，显式 ID 未命中不能退回模糊检索。
        exact_id = spec.sku_id or spec.product_id
        identifiers = [exact_id] if exact_id else list(dict.fromkeys(re.findall(r"(?<![A-Za-z0-9])P\d{4,}(?:-S\d+)?(?![A-Za-z0-9-])", spec.normalized_query.upper())))
        if identifiers:
            return await self._execute_exact_ids(spec, identifiers)
        if self._hybrid_enabled:
            return await self._execute_hybrid(spec)
        scored: list[tuple[float, Product]] = []
        recall_strategy = "keyword_2gram"
        rerank_applied = False

        if self._embedder is not None and self._vector_index is not None:
            try:
                scored = await self._vector_recall(spec)
                recall_strategy = "embedding_only"
            except Exception as err:  # noqa: BLE001 —— 召回基建异常必须降级而非失败
                logger.warning("向量召回不可用，降级关键词召回：%s", err)
                scored = []

        if not scored:
            scored = await self._keyword_recall(spec)
            recall_strategy = "keyword_2gram"

        candidates, filtered_out, diagnostics = self._prepare_candidates(spec, scored)
        if recall_strategy == "embedding_only" and candidates and self._reranker is not None:
            # 只有经过硬过滤与粗排收敛的候选进入精排。
            try:
                candidates = await self._rerank(spec, candidates)
                recall_strategy = "embedding_rerank"
                rerank_applied = True
            except Exception as err:  # noqa: BLE001
                logger.warning("rerank 不可用，按向量分排序：%s", err)
        hits = [self._candidate_card(score, product, spec) for score, product in candidates[: spec.top_k]]
        result = {
            "hits": hits,
            "total_candidates": diagnostics["eligible_candidates"],
            "recall_strategy": recall_strategy,
            "rerank_applied": rerank_applied,
            "retrieval_diagnostics": diagnostics,
        }
        if filtered_out:
            # 如实告知"召回到了但被硬约束挡掉"，否则模型分不清"库里没有"与"被过滤"，
            # 会把超预算商品答成"没有这个商品"
            result["filtered_out"] = filtered_out
        return result

    def _candidate_limit(self, spec: ProductSearchSpec) -> int:
        # 最终返回数量与精排容量分开；请求50项时不能只准备默认32项。
        return max(self._recall_candidates, spec.top_k)

    def _filter_candidates(self, spec: ProductSearchSpec, ranking):
        """两条检索路径共用硬过滤与拒绝摘要；不将软偏好变为硬条件。"""
        eligible, rejected = [], []
        for score, product in ranking:
            if self.eligible_skus(product, spec):
                eligible.append((score, product))
            elif len(rejected) < _FILTERED_OUT_LIMIT:
                rejected.append(self._rejected_product(product, spec, product.skus))
        return eligible, rejected

    def _prepare_candidates(self, spec: ProductSearchSpec, ranking):
        eligible, rejected = self._filter_candidates(spec, ranking)
        eligible.sort(key=lambda pair: pair[0], reverse=True)
        limit = self._candidate_limit(spec)
        pool = eligible[:limit]
        return pool, rejected, {"recalled_candidates": len(ranking),
            "eligible_candidates": len(eligible), "candidate_limit": limit,
            "rerank_candidates": len(pool)}

    async def _execute_exact_ids(self, spec: ProductSearchSpec, identifiers: list[str]) -> dict:
        product_ids = list(dict.fromkeys(identifier.split("-S", 1)[0] for identifier in identifiers))
        products = {p.product_id: p for p in await self._product_repo.find_by_ids(product_ids)}
        hits, rejected, missing = [], [], []
        for product_id in product_ids:
            product = products.get(product_id)
            requested = [identifier for identifier in identifiers if identifier.split("-S", 1)[0] == product_id]
            if product is None:
                missing.extend(requested)
                continue
            specific = [identifier for identifier in requested if "-S" in identifier]
            # 同一商品出现明确规格后，不能再因泛商品 ID 而回退默认规格。
            selected = specific or [sku.sku_id for sku in product.skus]
            eligible_skus = []
            for sku_id in selected:
                sku = product.find_sku(sku_id)
                if sku is None:
                    missing.append(sku_id)
                    continue
                if not self.sku_constraint_issues(product, sku, spec):
                    eligible_skus.append(sku)
                elif specific:
                    rejected.append(self._rejected_product(product, spec, [sku]))
            if eligible_skus:
                eligible_skus = self._sort_skus(eligible_skus, spec)
                card = self.product_card(1.0, product, spec, primary=eligible_skus[0], skus=eligible_skus).to_dict()
                hits.append(card)
            elif not specific:
                rejected.append(self._rejected_product(product, spec, product.skus))
        return {"hits": hits[:spec.top_k], "total_candidates": len(hits),
                "recall_strategy": "exact_id_lookup", "rerank_applied": False,
                "requested_identifiers": identifiers, "missing_identifiers": missing,
                "filtered_out": rejected[:_FILTERED_OUT_LIMIT], "existence_checked": True}

    async def _execute_hybrid(self, spec: ProductSearchSpec) -> dict:
        # 两路共享同一份权威目录快照；过滤资格不依赖向量库的陈旧价格或库存。
        products = await self._product_repo.list_all()
        permitted = [p.product_id for p in products if self.eligible_skus(p, spec)]
        diagnostics = {"filter_mode": "adaptive_post_filter", "eligible_products": len(permitted),
                       "lexical_weight": self._fusion_weights[0], "vector_weight": self._fusion_weights[1],
                       "candidate_limit": self._candidate_limit(spec)}
        async def lexical():
            return bm25_rank(spec.normalized_query, products)
        async def vector():
            if self._embedder is None or self._vector_index is None:
                return None
            try:
                embedding = await self._embedder.embed(spec.normalized_query)
                filtered_search = getattr(self._vector_index, "search_filtered", None)
                if filtered_search is not None:
                    hits = await filtered_search(embedding, top_n=diagnostics["candidate_limit"], product_ids=permitted)
                    if hits is not None:
                        diagnostics["filter_mode"] = "authoritative_ids"
                        by_id = {p.product_id: p for p in products}
                        return [(h.score, by_id[h.product_id]) for h in hits if h.product_id in by_id]
                return await self._vector_recall(spec, embedding=embedding)
            except Exception as err:
                logger.warning("Hybrid 向量侧不可用：%s", type(err).__name__)
                return None
        lexical_hits, vector_hits = await asyncio.gather(lexical(), vector())
        # 硬约束先于候选截断，BM25会扫描当前小目录的全部词项命中。
        lexical_hits, lexical_rejected = self._filter_candidates(spec, lexical_hits)
        lexical_hits = lexical_hits[:diagnostics["candidate_limit"]]
        vector_eligible, vector_rejected = self._filter_candidates(spec, vector_hits or [])
        rejected = (lexical_rejected + vector_rejected)[:_FILTERED_OUT_LIMIT]
        # 单路故障时保留健康侧，不让零权重把降级结果清空。
        weights = self._fusion_weights if vector_hits is not None else (1.0, 0.0)
        scored = reciprocal_rank_fusion(lexical_hits, vector_eligible, weights=weights)
        stages = None
        if self._capture_retrieval_stages:
            stages = {
                "lexical_candidates": [p.product_id for _, p in lexical_hits],
                "vector_candidates": [p.product_id for _, p in vector_eligible],
                # 零权重的路不参与融合；向量故障时以实际降级权重为准。
                "merged_candidates": list(dict.fromkeys(
                    p.product_id for ranking, weight in zip((lexical_hits, vector_eligible), weights)
                    if weight > 0 for _, p in ranking)),
                "fused_candidates": [p.product_id for _, p in scored],
            }
        scored, pool_rejected, pool_diagnostics = self._prepare_candidates(spec, scored)
        rejected = (rejected + pool_rejected)[:_FILTERED_OUT_LIMIT]
        diagnostics.update(pool_diagnostics)
        if stages is not None:
            stages["rerank_candidates"] = [p.product_id for _, p in scored]
        strategy = "hybrid_only" if vector_hits is not None else "bm25"
        rerank_applied = False
        if scored and self._reranker is not None:
            try:
                scored = await self._rerank(spec, scored)
                strategy, rerank_applied = ("hybrid_rerank" if vector_hits is not None else "bm25_rerank"), True
            except Exception as err:
                diagnostics["reranker_error"] = getattr(err, "code", type(err).__name__)
                logger.warning("Hybrid 重排不可用：%s", diagnostics["reranker_error"])
        deduped, seen = [], set()
        for score, product in scored:
            key = product.canonical_product_id or product.product_id
            if key not in seen:
                seen.add(key)
                deduped.append((score, product))
        result = {"hits": [self._candidate_card(score, p, spec) for score, p in deduped[:spec.top_k]],
                "total_candidates": len(deduped), "recall_strategy": strategy, "rerank_applied": rerank_applied,
                "retrieval_variant": "bm25_vector_rrf_v1", "vector_available": vector_hits is not None,
                "filtered_out": rejected, "retrieval_diagnostics": diagnostics}
        if stages is not None:
            stages["ranked_candidates"] = [p.product_id for _, p in deduped]
            result["retrieval_stages"] = stages
        return result

    async def product_details(self, product_id: str, spec: ProductSearchSpec, *, sku_id=None, landed_budget_major=None) -> dict:
        """指定对象读取不筛掉商品；当前条件只用于解释，不能改变读取权限或购物状态。"""
        product = await self._product_repo.find_by_id(product_id)
        if product is None:
            return {"hits": [], "missing_identifiers": [product_id], "existence_checked": True}
        primary = product.find_sku(sku_id) if sku_id else product.primary_available_sku()
        if primary is None:
            return {"hits": [], "missing_identifiers": [sku_id], "existence_checked": True}
        # 装配完整资料时不传排除规格，不能让当前条件裁掉用户想查看的规格。
        display = ProductSearchSpec(product_id=product_id, target_currency=spec.target_currency, ship_to=spec.ship_to)
        card = self.product_card(1.0, product, display, primary=primary, skus=product.skus).to_dict()
        for raw, sku in zip(card["skus"], product.skus):
            converted = self._tariff.rates.convert(sku.price, spec.target_currency)
            raw["display_price_major"] = converted.to_major_units()
            raw["display_currency"] = spec.target_currency
            raw["constraint_issues"] = self.sku_constraint_issues(product, sku, spec)
            if landed_budget_major is not None and spec.ship_to:
                try:
                    line = self.pricing.price_sku(product, sku, 1, spec.ship_to, spec.target_currency)
                    if line["total_amount_minor"] > round(landed_budget_major * 100):
                        raw["constraint_issues"].append("over_landed_budget")
                except ValueError:
                    # 不可配送已由约束解释；详情仍返回，不伪造可购买报价。
                    pass
        return {"hits": [card], "missing_identifiers": [], "existence_checked": True, "purpose": "product_details"}

    def sku_constraint_issues(self, product: Product, sku: Sku, spec: ProductSearchSpec) -> list[str]:
        """搜索、推荐及详情说明共用逐 SKU 资格；不同规格不能互借价格或库存。"""
        issues = []
        if product.product_id in spec.excluded_product_ids:
            issues.append("product_excluded")
        if sku.sku_id in spec.excluded_sku_ids:
            issues.append("sku_excluded")
        if sku.stock <= 0:
            issues.append("out_of_stock")
        if spec.category and product.category != spec.category:
            issues.append("category_mismatch")
        if set(product.material_tags) & (set(spec.excluded_material_tags) | set(spec.excluded_materials_by_category.get(product.category, []))):
            issues.append("material_excluded")
        if set(spec.required_material_tags) - set(product.material_tags):
            issues.append("material_required_missing")
        if spec.ship_to and spec.ship_to not in product.ships_to:
            issues.append("ship_to_unavailable")
        price = self._tariff.rates.convert(sku.price, spec.target_currency)
        if spec.price_max_major is not None and price.to_major_units() > spec.price_max_major:
            issues.append("over_price_cap")
        return issues

    def _sort_skus(self, skus: list[Sku], spec: ProductSearchSpec) -> list[Sku]:
        return sorted(skus, key=lambda sku: (
            self._tariff.rates.convert(sku.price, spec.target_currency).amount_in_minor_units, sku.sku_id))

    def eligible_skus(self, product: Product, spec: ProductSearchSpec) -> list[Sku]:
        """返回同时满足本次全部硬条件的规格；顺序不依赖目录中 SKU 的排列。"""
        return self._sort_skus([sku for sku in product.skus
            if (spec.sku_id is None or sku.sku_id == spec.sku_id)
            and not self.sku_constraint_issues(product, sku, spec)], spec)

    def _rejected_product(self, product: Product, spec: ProductSearchSpec, skus: list[Sku]) -> dict:
        rows = [{"sku_id": sku.sku_id, "spec": sku.spec,
                 "price_major": self._tariff.rates.convert(sku.price, spec.target_currency).to_major_units(),
                 "currency": spec.target_currency,
                 "reasons": self.sku_constraint_issues(product, sku, spec)}
                for sku in self._sort_skus(skus, spec)]
        first_reasons = {row["reasons"][0] for row in rows}
        return {
            "product_id": product.product_id,
            "title": product.title,
            "category": product.category,
            "reason": next(iter(first_reasons)) if len(first_reasons) == 1 else "no_eligible_sku",
            "skus": rows,
        }

    def _candidate_card(self, score: float, product: Product, spec: ProductSearchSpec) -> dict:
        skus = self.eligible_skus(product, spec)
        return self.product_card(score, product, spec, primary=skus[0], skus=skus).to_dict()

    # ---- 一阶段：向量召回 ----

    async def _vector_recall(self, spec: ProductSearchSpec, embedding=None) -> list[tuple[float, Product]]:
        if embedding is None:
            embedding = await self._embedder.embed(spec.normalized_query)
        limit = self._candidate_limit(spec)
        top_n = limit
        while True:
            vector_hits = await self._vector_index.search(embedding, top_n=top_n)
            products = await self._product_repo.find_by_ids([hit.product_id for hit in vector_hits])
            by_id = {product.product_id: product for product in products}
            scored = [(hit.score, by_id[hit.product_id]) for hit in vector_hits if hit.product_id in by_id]
            if len(vector_hits) < top_n or top_n >= _MAX_VECTOR_RECALL or sum(bool(self.eligible_skus(p, spec)) for _, p in scored) >= limit:
                return scored
            top_n = min(_MAX_VECTOR_RECALL, top_n*2)

    # ---- 二阶段：精排 ----

    async def _rerank(
        self,
        spec: ProductSearchSpec,
        scored: list[tuple[float, Product]],
    ) -> list[tuple[float, Product]]:
        if self._reranker is None:
            raise RuntimeError("Reranker 未配置")
        documents = [product.searchable_text(skus=self.eligible_skus(product, spec)) for _, product in scored]
        rerank_scores = await self._reranker.rerank(spec.normalized_query, documents)
        import math
        if len(rerank_scores) != len(scored) or not all(math.isfinite(float(score)) for score in rerank_scores):
            raise ValueError("重排分数必须等长且有限")
        reranked = [
            (rerank_scores[i], product)
            for i, (_, product) in enumerate(scored)
        ]
        reranked.sort(key=lambda pair: pair[0], reverse=True)
        return reranked

    # ---- 兜底：关键词召回 ----

    async def _keyword_recall(self, spec: ProductSearchSpec) -> list[tuple[float, Product]]:
        query_terms = tokenize(spec.normalized_query)
        candidates: list[tuple[float, Product]] = []
        for product in await self._product_repo.list_all():
            score = self._keyword_score(query_terms, product, spec)
            if score > 0:
                candidates.append((score, product))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return candidates

    @staticmethod
    def _keyword_score(query_terms: set[str], product: Product, spec: ProductSearchSpec) -> float:
        doc_terms = tokenize(product.searchable_text())
        matched = query_terms & doc_terms
        if not matched:
            return 0.0
        score = float(len(matched))
        # 品类槽位命中加权，让"槽位过滤"优于全文命中
        if spec.category and spec.category in product.category:
            score += 3.0
        return score

    # ---- 商品卡组装（含到手价内联）----

    def product_card(self, score: float, product: Product, spec: ProductSearchSpec, *,
                     primary: Sku, skus: list[Sku]) -> ProductCard:
        """调用方明确传入本次规格集合；详情、候选、最终选择各自按用途提供。"""
        media = product_media(product.product_id, product.title)
        primary_in_target = self._tariff.rates.convert(primary.price, spec.target_currency)
        landed_price: Optional[dict] = None
        if spec.ship_to:
            try:
                line = self.pricing.price_sku(product, primary, 1, spec.ship_to, spec.target_currency)
                landed_price = self.pricing.assemble([line], spec.ship_to, spec.target_currency)
            except ValueError as err:
                # 目的国不在规则表内：如实标注，不编造数字
                landed_price = {"unavailable_reason": str(err)}
        return ProductCard(
            product_id=product.product_id,
            title=product.title,
            brand=product.brand,
            category=product.category,
            origin_country=product.origin_country,
            # 商品卡价格必须与预算过滤使用同一目标币种；原始报价单独保留，
            # SKU 详情继续维持平台原币种，方便后续下单时选择具体规格。
            price_major=primary_in_target.to_major_units(),
            currency=spec.target_currency,
            source_price_major=primary.price.to_major_units(),
            source_currency=primary.price.currency,
            highlights=[f"{h.label}：{h.detail}" if h.detail else h.label for h in product.highlights],
            skus=[
                {
                    "sku_id": sku.sku_id,
                    "spec": sku.spec,
                    "price_major": sku.price.to_major_units(),
                    "currency": sku.price.currency,
                    "stock": sku.stock,
                }
                for sku in skus
            ],
            score=score,
            landed_price=landed_price,
            source_platform=product.source_platform,
            canonical_product_id=product.canonical_product_id,
            material_tags=product.material_tags,
            weight_kg=product.weight_kg,
            description=product.description,
            rating_summary=dict(product.rating_summary) if product.rating_summary is not None else None,
            ships_to=list(product.ships_to),
            dimensions_cm=dict(product.dimensions_cm),
            package_dimensions_cm=dict(product.package_dimensions_cm),
            updated_at=product.updated_at,
            source_language=product.source_language,
            source_locale=product.source_locale,
            data_provenance=product.data_provenance,
            default_sku_id=primary.sku_id,
            image_url=media.image_url,
            image_kind=media.image_kind,
            image_alt=media.image_alt,
        )
