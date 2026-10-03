"""Skill 唯一只读入口：保留目录版本、买家归属和角色工具范围校验。"""
from app.infrastructure.context import ShoppingContext
from app.infrastructure.capability_registry import SKILL_TOOL_ALLOWLIST


def read_skill(registry, personal_store, available_tools, skill_id, version, *, expected_digest=None):
    context = ShoppingContext.current()
    if context is None or registry is None:
        raise ValueError("Skill 读取缺少可信会话")
    digest = expected_digest if expected_digest is not None else registry.bind_session(
        context.shopping_session_id, context.buyer_id,
        allow_skill_updates=context.skill_catalog_mode == "append_only")
    if context.capability_digest and digest != context.capability_digest:
        raise ValueError("本轮 Skill 资料版本变化，请重试")
    tools = frozenset(available_tools) & SKILL_TOOL_ALLOWLIST
    if skill_id.startswith("personal-"):
        if personal_store is None:
            raise ValueError("个人 Skill 存储不可用")
        loaded = personal_store.load(context.buyer_id, skill_id, version)
        if not set(loaded["allowed_tools"]) <= tools:
            raise ValueError("Skill 所需工具不适用于当前 Agent")
        return loaded
    return registry.load_skill(skill_id, version, available_tools=tools,
                               expected_digest=digest, require_current=True)
