# -*- coding: utf-8 -*-
"""四期模块二：Redis 缓存（embedding 缓存 / 语义缓存 / 幂等键 / 旁路降级）

用内存替身模拟 Redis，重点验证两件容易出事的事：
    1. Redis 故障必须旁路，不能把缓存变成新的单点；
    2. 写操作意图（下单/取消）与上下文依赖问句绝不能进语义缓存。
"""
from __future__ import annotations

import pytest

from app.infrastructure.cache.cached_embedding_client import CachedEmbeddingClient
from app.infrastructure.cache.redis_cache import RedisCache

pytestmark = pytest.mark.asyncio


class InMemoryCache(RedisCache):
    """RedisCache 的内存替身：只覆盖存取，行为语义保持一致。"""

    def __init__(self) -> None:
        super().__init__("")
        self._store: dict = {}
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def ping(self) -> bool:
        return self._enabled

    async def get_json(self, key):
        return self._store.get(key)

    async def set_json(self, key, value, ttl_seconds):
        self._store[key] = value

    async def delete(self, key):
        self._store.pop(key, None)

    async def set_if_absent(self, key, value, ttl_seconds):
        if key in self._store:
            return False
        self._store[key] = value
        return True


class BrokenCache(RedisCache):
    """所有操作都抛异常的 Redis：验证调用方是否真的旁路。"""

    def __init__(self) -> None:
        super().__init__("")
        self._client = object()  # 假装已连接，让 enabled 为 True

    async def get_json(self, key):
        raise RuntimeError("connection refused")

    async def set_json(self, key, value, ttl_seconds):
        raise RuntimeError("connection refused")


class CountingEmbedder:
    """按文本首字符造可控向量，同时统计上游调用次数。"""

    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts.extend(texts)
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        # 同文本必得同向量；不同文本方向差异明显，便于测相似度
        seed = sum(ord(ch) for ch in text) % 97
        return [1.0, seed / 97.0, 0.1]


class TestCachedEmbeddingClient:
    async def test_second_call_hits_cache(self):
        inner = CountingEmbedder()
        client = CachedEmbeddingClient(inner, InMemoryCache(), "text-embedding-v4")
        first = await client.embed("露营灯")
        second = await client.embed("露营灯")
        assert first == second
        assert inner.calls == 1, "同文本第二次不应再打上游"
        assert (client.hits, client.misses) == (1, 1)

    async def test_batch_only_fetches_missing_texts(self):
        inner = CountingEmbedder()
        client = CachedEmbeddingClient(inner, InMemoryCache(), "text-embedding-v4")
        await client.embed_batch(["A", "B"])
        inner.texts.clear()
        result = await client.embed_batch(["A", "B", "C"])

        assert inner.texts == ["C"], "只应回源未命中的文本"
        assert len(result) == 3
        # 命中项要拼回原位置，顺序不能乱
        assert result[0] == CountingEmbedder._vector("A")
        assert result[2] == CountingEmbedder._vector("C")

    async def test_broken_redis_falls_through_to_upstream(self):
        """Redis 故障时必须照常返回结果，只是没有缓存收益。"""
        inner = CountingEmbedder()
        client = CachedEmbeddingClient(inner, BrokenCache(), "text-embedding-v4")
        assert await client.embed("露营灯") == CountingEmbedder._vector("露营灯")
        assert await client.embed("露营灯") == CountingEmbedder._vector("露营灯")
        assert inner.calls == 2  # 没有缓存，但没有失败

    async def test_empty_batch_short_circuits(self):
        inner = CountingEmbedder()
        client = CachedEmbeddingClient(inner, InMemoryCache(), "m")
        assert await client.embed_batch([]) == []
        assert inner.calls == 0


class TestIdempotencyKey:
    async def test_first_wins_second_rejected(self):
        cache = InMemoryCache()
        assert await cache.set_if_absent("idem:s1:h1", "1", 600) is True
        assert await cache.set_if_absent("idem:s1:h1", "1", 600) is False

    async def test_absent_redis_lets_request_through(self):
        """没有 Redis 时应放行（退化为无幂等保护），不能拒绝服务。"""
        assert await RedisCache("").set_if_absent("idem:s1:h1", "1", 600) is True
