# -*- coding: utf-8 -*-
"""Markdown 品类知识库。

在线知识检索使用独立 Qdrant 集合；确定性词项实现用于离线诊断及显式降级。
"""
from __future__ import annotations

import json
import hashlib
import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from app.infrastructure.settings import PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

_KB_DESCRIPTION = (
    "Globex 跨境电商品类洞察知识库：各品类的热卖款型、关键属性判断口径、"
    "价格区间参考、避坑点，以及跨境到手价/免税额度/合规通则。"
)

# 以 Qdrant cosine similarity 为口径；低于此值的“最近邻”只是被迫返回的噪声，
# 必须显式拒答，不能包装成知识库事实。阈值由正式不可回答集校准并以回归测试固定。
MIN_ANSWERABLE_KNOWLEDGE_SCORE = 0.20


@dataclass
class KnowledgeChunk:
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class KnowledgeDocument:
    document_id: str
    metadata: dict
    chunks: list[KnowledgeChunk] = field(default_factory=list)


@dataclass
class KnowledgeResult:
    document_id: str
    chunk: KnowledgeChunk
    score: float


class MarkdownKnowledgeBase:
    """小规模版本化知识目录的框架无关实现。"""

    def __init__(self, name: str, description: str, collection: str):
        self.name, self.description, self.collection = name, description, collection
        self._documents: dict[str, KnowledgeDocument] = {}

    async def ensure_collection(self) -> None:
        return None

    async def list_documents(self) -> list[KnowledgeDocument]:
        return list(self._documents.values())

    async def delete_document(self, document_id: str) -> None:
        self._documents.pop(document_id, None)

    async def insert_document(self, *, chunks, document_id: str, document_metadata: dict) -> None:
        normalized = [
            chunk if isinstance(chunk, KnowledgeChunk) else KnowledgeChunk(str(getattr(chunk, "content", chunk)), dict(getattr(chunk, "metadata", {})))
            for chunk in chunks
        ]
        self._documents[document_id] = KnowledgeDocument(document_id, dict(document_metadata), normalized)

    async def search(self, *, queries: list[str], top_k: int = 3) -> list[KnowledgeResult]:
        terms = set().union(*(_lexical_terms(query) for query in queries))
        ranked: list[KnowledgeResult] = []
        for document in self._documents.values():
            for chunk in document.chunks:
                overlap = terms & _lexical_terms(chunk.content)
                if overlap:
                    score = len(overlap) / max(len(terms), 1)
                    ranked.append(KnowledgeResult(document.document_id, chunk, score))
        ranked.sort(key=lambda item: (-item.score, item.document_id))
        return ranked[:top_k]


def has_answerable_knowledge(results, min_score: float = MIN_ANSWERABLE_KNOWLEDGE_SCORE) -> bool:
    """判断检索结果是否达到可回答的最低相关性，缺少 score 一律拒答。"""
    scores: list[float] = []
    for item in results:
        try:
            scores.append(float(getattr(item, "score")))
        except (TypeError, ValueError, AttributeError):
            continue
    return bool(scores) and max(scores) >= min_score


def policy_fact_status(metadata: dict, today: date | None = None) -> str:
    """判定政策知识能否被当作确定事实引用。

    非政策类只是选购参考，不套用法规口径。政策类必须来自权威快照、已生效且未过期；
    合成评测资料会明确返回不可作确定事实，避免模型把演示文档包装成法规结论。
    """
    if metadata.get("topic") != "policy":
        return "not_policy"
    if not metadata.get("source_reference"):
        return "missing_source"
    if metadata.get("source_type") != "official_snapshot":
        return "non_authoritative_source"
    today = today or date.today()
    try:
        effective_from = date.fromisoformat(str(metadata.get("effective_from", "")))
        effective_to = date.fromisoformat(str(metadata.get("effective_to", "")))
    except ValueError:
        return "invalid_effective_date"
    if today < effective_from:
        return "not_effective"
    if today > effective_to:
        return "expired"
    return "fact_eligible"


def load_knowledge_metadata(knowledge_dir: Path = KNOWLEDGE_DIR) -> dict[str, dict]:
    """读取 manifest，把来源/时效元数据随 chunk 一起入库。

    ``source`` 保留文件名以兼容既有召回评测；实际来源放在 ``source_reference``，
    防止“评测标注单位”与“引用来源”混为一谈。
    """
    manifest = knowledge_dir / "manifest.jsonl"
    if not manifest.exists():
        # 临时知识库（单测/本地实验）仍应可启动；正式知识库是否缺 manifest
        # 由 scripts/eval/knowledge_quality.py 的数据门禁负责阻断。
        return {
            path.stem: {
                "source": path.name,
                "source_reference": "未登记的临时知识文档",
                "source_type": "unversioned_fixture",
                "published_at": "1970-01-01",
                "effective_from": "1970-01-01",
                "effective_to": "1970-01-01",
                "region": "GLOBAL",
                "version": "unversioned",
                "topic": "unknown",
            }
            for path in knowledge_dir.glob("*.md")
        }
    metadata: dict[str, dict] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        metadata[entry["document_id"]] = {
            "source": entry["filename"],
            "source_reference": entry["source"],
            "source_type": entry["source_type"],
            "published_at": entry["published_at"],
            "effective_from": entry["effective_from"],
            "effective_to": entry["effective_to"],
            "region": entry["region"],
            "version": entry["version"],
            "topic": entry["topic"],
        }
    return metadata


def _lexical_terms(text: str) -> set[str]:
    """生成适合中文短查询的确定性词项；仅用于向量服务故障时降级。"""
    normalized = text.lower()
    if "多少钱" in normalized or "合理" in normalized:
        normalized += " 价格 预算 区间"
    if "怎么挑" in normalized or "怎么选" in normalized:
        normalized += " 选购 判断 避坑"
    if "自重" in normalized:
        normalized += " 重量 轻量"
    terms = set(re.findall(r"[a-z0-9_]+", normalized))
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(sequence) == 1:
            terms.add(sequence)
        else:
            terms.update(sequence[index:index + 2] for index in range(len(sequence) - 1))
    return terms


def keyword_fallback_insights(
    question: str,
    knowledge_dir: Path = KNOWLEDGE_DIR,
    top_k: int = 3,
) -> list[dict]:
    """从版本化 Markdown 中按段落检索，作为 embedding 异常时的可追溯降级。"""
    query_terms = _lexical_terms(question)
    metadata_by_id = load_knowledge_metadata(knowledge_dir)
    candidates: list[tuple[int, str, int, str, dict]] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+.+$", text, flags=re.MULTILINE)
        title = title_match.group(0) if title_match else f"# {path.stem}"
        sections = [section.strip() for section in re.split(r"(?=^##\s+)", text, flags=re.MULTILINE) if section.strip()]
        for index, section in enumerate(sections):
            content = section if section.startswith(title) else f"{title}\n\n{section}"
            overlap = query_terms & _lexical_terms(content)
            if not overlap:
                continue
            candidates.append((len(overlap), path.stem, index, content, metadata_by_id.get(path.stem, {})))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    insights: list[dict] = []
    for score, document_id, _, content, metadata in candidates[:max(1, top_k)]:
        insights.append({
            "content": content,
            "source": metadata.get("source", f"{document_id}.md"),
            "score": round(score / max(len(query_terms), 1), 4),
            "metadata": {
                key: metadata[key]
                for key in ("source_reference", "source_type", "published_at", "effective_from", "effective_to", "region", "version", "topic")
                if key in metadata
            },
        })
    return insights


def build_category_knowledge_base(settings: Settings):
    """在线使用 embedding + Qdrant；不可用时由工具明确标记词项降级。"""
    from app.infrastructure.rag.vector_knowledge import VectorKnowledgeBase
    return VectorKnowledgeBase(settings)


async def bootstrap_category_knowledge(
    knowledge_base,
    knowledge_dir: Optional[Path] = None,
) -> int:
    """把 knowledge/*.md 灌入知识库（幂等），返回入库文档数；失败仅告警返回 0。"""
    directory = knowledge_dir or KNOWLEDGE_DIR
    try:
        metadata_by_id = load_knowledge_metadata(directory)
        await knowledge_base.ensure_collection()
        existing = {doc.document_id: doc for doc in await knowledge_base.list_documents()}
        local_ids = {path.stem for path in directory.glob("*.md")}
        # 此 collection 专用于该知识目录；目录缺失时拒绝执行删除，避免路径配置错误清库。
        if not directory.is_dir():
            raise ValueError("知识目录不存在，拒绝同步")
        for document_id, doc in existing.items():
            if document_id not in local_ids and doc.metadata.get("managed_by") == "globex_markdown_sync_v1":
                await knowledge_base.delete_document(document_id)
        inserted = 0
        for md_file in sorted(directory.glob("*.md")):
            document_id = md_file.stem
            metadata = metadata_by_id[document_id]
            content = md_file.read_text(encoding="utf-8")
            title_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
            document_title = title_match.group(1).strip() if title_match else document_id
            metadata = {**metadata, "document_title": document_title}
            content_hash = hashlib.sha256(md_file.read_bytes() + b"\0" + json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode() + b"\0chunker:chars1536:overlap100:v1").hexdigest()
            previous = existing.get(document_id)
            if previous is not None and previous.metadata.get("content_sha256") == content_hash:
                continue
            sections = [part.strip() for part in re.split(r"(?=^##\s+)", content, flags=re.MULTILINE) if part.strip()]
            chunks = [KnowledgeChunk(section[start:start + 1536],
                {**metadata, "content_sha256": content_hash, "managed_by": "globex_markdown_sync_v1"})
                for section in sections for start in range(0, len(section), 1436)]
            # SDK 不支持原子换版。先删旧文档再插新版本；失败时允许暂时缺失，不能继续引用过期知识。
            if previous is not None:
                await knowledge_base.delete_document(document_id)
            await knowledge_base.insert_document(
                chunks=chunks,
                document_id=document_id,
                document_metadata={**metadata, "content_sha256": content_hash, "managed_by": "globex_markdown_sync_v1"},
            )
            inserted += 1
        logger.info(
            "品类知识库就绪：写入或更新 %d 篇，累计 %d 篇",
            inserted,
            len(await knowledge_base.list_documents()),
        )
        return inserted
    except Exception as err:  # noqa: BLE001 —— 知识库不可用不阻塞启动
        logger.warning("品类知识库建库失败，category_insight 将不可用：%s", err)
        return 0
