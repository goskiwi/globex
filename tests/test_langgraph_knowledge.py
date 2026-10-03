"""真实本地 Qdrant 验证语义路径、重启和知识换版；仅 embedding 使用确定性替身。"""
from types import SimpleNamespace
from app.infrastructure.rag.vector_knowledge import VectorKnowledgeBase
from app.infrastructure.rag.category_knowledge import bootstrap_category_knowledge
from app.infrastructure.rag.knowledge_retrieval import search_knowledge


class Embeddings:
    def __init__(self):
        self.calls = 0

    async def embed_batch(self, texts):
        self.calls += 1
        return [[1.0, 0.0] if "灯" in text or "夜行" in text else [0.0, 1.0] for text in texts]


async def test_vector_knowledge_survives_restart_and_replaces_deleted_documents(tmp_path):
    directory = tmp_path / "docs"
    directory.mkdir()
    light = directory / "light.md"
    light.write_text("# 照明指南\n露营灯用于营地照明。")
    cup = directory / "cup.md"
    cup.write_text("# 饮具指南\n陶瓷杯适合冲泡饮品。")
    settings = SimpleNamespace(category_kb_collection="kb", embedding_dim=2,
                               data_dir=tmp_path, qdrant_url="")
    embeddings = Embeddings()
    kb = VectorKnowledgeBase(settings, embedder=embeddings)
    try:
        assert await bootstrap_category_knowledge(kb, directory) == 2
        found = await search_knowledge(kb, "夜行", 1)
        assert found[0].document_id == "light" and found[0].score > .99
        calls = embeddings.calls
        assert await bootstrap_category_knowledge(kb, directory) == 0
        assert embeddings.calls == calls
    finally:
        await kb.close()
    kb = VectorKnowledgeBase(settings, embedder=embeddings)
    try:
        assert len(await kb.list_documents()) == 2
        light.write_text("# 照明指南\n露营灯需要核验防水参数。")
        cup.unlink()
        assert await bootstrap_category_knowledge(kb, directory) == 1
        documents = await kb.list_documents()
        assert [document.document_id for document in documents] == ["light"]
        assert "防水" in (await search_knowledge(kb, "夜行", 1))[0].chunk.content
        assert "营地照明" not in documents[0].chunks[0].content
    finally:
        await kb.close()
