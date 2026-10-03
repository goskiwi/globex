"""使用项目现有 HTTP 客户端验证 BGE 配置；在 Docker 中执行，不打印密钥。"""
import asyncio
import json
import math
import time
from app.infrastructure.settings import load_settings
from app.infrastructure.embedding.openai_embedding_client import OpenAIEmbeddingClient
from app.infrastructure.rerank.factory import create_reranker


async def main():
    settings=load_settings()
    assert settings.embedding_model=='BAAI/bge-m3'
    assert settings.reranker_model=='BAAI/bge-reranker-v2-m3'
    embedder=OpenAIEmbeddingClient(settings)
    reranker=create_reranker(settings)
    assert reranker is not None
    query='适合旅行的防水背包'
    documents=['Waterproof travel backpack with comfortable straps.','A stainless steel electric kettle.','Waterproof travel backpack with comfortable straps.']
    started=time.monotonic()
    vectors=await embedder.embed_batch([query,*documents])
    assert len(vectors)==4 and all(len(v)==1024 and all(math.isfinite(x) for x in v) for v in vectors)
    norms=[sum(x*x for x in v)**.5 for v in vectors]
    assert all(abs(n-1)<.02 for n in norms)
    embedding_ms=(time.monotonic()-started)*1000
    started=time.monotonic()
    scores=await reranker.rerank(query,documents)
    assert len(scores)==3 and scores[0]==scores[2] and scores[0]>scores[1]
    print(json.dumps({'status':'passed','embedding_model':settings.embedding_model,'reranker_model':settings.reranker_model,
        'embedding_dimension':1024,'embedding_norms':norms,'reranker_scores':scores,
        'duplicate_positions_preserved':True,'embedding_ms':round(embedding_ms,2),
        'reranker_ms':round((time.monotonic()-started)*1000,2)},ensure_ascii=False))


if __name__=='__main__':asyncio.run(main())
