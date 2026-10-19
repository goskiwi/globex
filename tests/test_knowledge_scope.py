"""适用范围在召回前生效；通用资料显式保留，未知范围不冒充相关。"""
from types import SimpleNamespace
from qdrant_client import AsyncQdrantClient
from app.infrastructure.rag.vector_knowledge import VectorKnowledgeBase
from app.infrastructure.rag.category_knowledge import KnowledgeChunk
from app.infrastructure.rag.knowledge_retrieval import KnowledgeScope,search_knowledge

class SameVector:
    async def embed_batch(self,texts):return [[1.,0.] for _ in texts]

async def test_scope_filters_equal_vectors_and_keeps_declared_general_knowledge(tmp_path):
    kb=VectorKnowledgeBase(SimpleNamespace(category_kb_collection='isolated',embedding_dim=2),client=AsyncQdrantClient(':memory:'),embedder=SameVector())
    try:
        await kb.ensure_collection()
        for identifier,category,topic,region in [('travel','旅行装备','category','GLOBAL'),('audio','数码配件','category','GLOBAL'),('global',None,'policy','GLOBAL'),('us',None,'policy','US'),('cn',None,'policy','CN'),('unknown',None,'category','GLOBAL')]:
            metadata={'topic':topic,'region':region,'applicability':{'categories':[category] if category else [],'all_categories':topic=='policy'}}
            await kb.insert_document(chunks=[KnowledgeChunk('通勤怎么挑',metadata)],document_id=identifier,document_metadata=metadata)
        rows=await search_knowledge(kb,'通勤怎么挑',10,scope=KnowledgeScope(topic='category',categories=['旅行装备']))
        assert [r.document_id for r in rows]==['travel']
        rows=await search_knowledge(kb,'通勤怎么挑',10,scope=KnowledgeScope(topic='category',categories=['旅行装备','数码配件']))
        assert {r.document_id for r in rows}=={'travel','audio'}
        rows=await search_knowledge(kb,'跨境规则',10,scope=KnowledgeScope(topic='policy',categories=[],region='CN'))
        assert {r.document_id for r in rows}=={'global','cn'}
        assert await search_knowledge(kb,'通勤怎么挑',3,scope=KnowledgeScope(topic='category',categories=['未登记品类']))==[]
    finally:await kb.close()
