"""框架无关的品类向量检索，复用项目 embedding 客户端与 Qdrant。"""
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from app.infrastructure.embedding.openai_embedding_client import OpenAIEmbeddingClient
from app.infrastructure.rag.category_knowledge import KnowledgeChunk, KnowledgeDocument, KnowledgeResult


class VectorKnowledgeBase:
    def __init__(self, settings, *, client=None, embedder=None):
        # 新格式使用独立集合；旧 SDK 集合保留用于回滚，不猜测或覆盖旧 payload。
        self.collection = settings.category_kb_collection + "_langchain_v1"
        self.dimension = settings.embedding_dim
        self.embedding_model = embedder or OpenAIEmbeddingClient(settings)
        self.client = client or (
            AsyncQdrantClient(url=settings.qdrant_url) if settings.qdrant_url
            else AsyncQdrantClient(path=str(settings.data_dir / "qdrant_kb_langchain"))
        )

    async def ensure_collection(self):
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(self.collection,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE))
        params = (await self.client.get_collection(self.collection)).config.params.vectors
        if not isinstance(params, VectorParams) or params.size != self.dimension or params.distance != Distance.COSINE:
            raise ValueError("知识向量维度或距离不匹配，请使用新的知识集合")

    async def list_documents(self):
        documents = {}
        offset = None
        while True:
            points, offset = await self.client.scroll(self.collection, offset=offset, limit=128,
                                                     with_payload=True, with_vectors=False)
            for point in points:
                payload = point.payload
                if not payload or "document_id" not in payload or "content" not in payload:
                    raise ValueError("知识集合包含不兼容记录，未覆盖旧数据")
                identifier = payload["document_id"]
                document = documents.setdefault(identifier, KnowledgeDocument(identifier, payload["metadata"], []))
                document.chunks.append(KnowledgeChunk(payload["content"], payload["metadata"]))
            if offset is None:
                return list(documents.values())

    @staticmethod
    def _filter(document_id):
        return Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])

    async def delete_document(self, document_id):
        await self.client.delete(self.collection, points_selector=self._filter(document_id), wait=True)

    async def insert_document(self, *, chunks, document_id, document_metadata):
        vectors = await self.embedding_model.embed_batch([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("知识片段与向量数量不一致")
        points = [PointStruct(id=str(uuid.uuid4()), vector=vector, payload={
            "document_id": document_id, "content": chunk.content,
            "metadata": {**document_metadata, **chunk.metadata},
        }) for chunk, vector in zip(chunks, vectors)]
        if points:
            await self.client.upsert(self.collection, points=points, wait=True)

    async def search(self, *, queries, top_k=3, document_id=None):
        vectors = await self.embedding_model.embed_batch(queries)
        results = []
        for vector in vectors:
            response = await self.client.query_points(
                self.collection, query=vector, limit=top_k, with_payload=True,
                query_filter=self._filter(document_id) if document_id else None,
            )
            results.extend(KnowledgeResult(
                point.payload["document_id"],
                KnowledgeChunk(point.payload["content"], point.payload["metadata"]),
                point.score,
            ) for point in response.points)
        return sorted(results, key=lambda item: -item.score)[:top_k]

    async def close(self):
        await self.client.close()
