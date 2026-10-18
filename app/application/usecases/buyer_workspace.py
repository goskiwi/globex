"""买家资料和可用流程目录；不依赖 Agent 工厂或会话内部结构。"""
import asyncio
from dataclasses import dataclass
from typing import Callable

from app.domain.buyer.preference import PreferenceStore


@dataclass
class BuyerWorkspace:
    skills: object
    preferences: PreferenceStore
    capabilities: object
    tool_names: Callable[[], set[str]]

    async def available_skills(self, buyer_id: str | None = None) -> dict:
        if self.capabilities is None:
            raise RuntimeError("选购方案服务尚未配置")
        available = self.tool_names()

        def read():
            digest = self.capabilities.version_fingerprint()
            metadata = self.capabilities.metadata(available_tools=available, expected_digest=digest)
            fields = ("id", "version", "title", "description", "scope", "content_hash", "expires_at")
            return {"capability_digest": digest, "skills": [{key: item[key] for key in fields} for item in metadata]}

        result = await asyncio.to_thread(read)
        if buyer_id and self.skills is not None:
            result["skills"] = [*await asyncio.to_thread(self.skills.list, buyer_id), *result["skills"]]
        return result
