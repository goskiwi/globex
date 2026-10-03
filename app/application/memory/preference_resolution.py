"""一次决策的权威偏好：召回只选展示项，不能决定硬约束是否存在。"""
from dataclasses import dataclass
from app.domain.buyer.preference import BuyerPreference
from app.application.memory.preference_selector import render_preference_hint


@dataclass(frozen=True)
class ResolvedPreferences:
    facts: tuple[BuyerPreference, ...]
    selected: tuple[BuyerPreference, ...]

    @property
    def hint(self):
        return (render_preference_hint(self.selected) if self.facts else
                "当前持久偏好：无。历史已撤回偏好不得恢复。")


async def resolve_preferences(store, selector, buyer_id, query, top_k):
    async def read():
        try:
            facts = tuple(await store.list_by_buyer(buyer_id))
            if any(p.buyer_id != buyer_id for p in facts):
                raise ValueError("偏好归属不一致")
            return facts
        except Exception as error:
            raise RuntimeError("当前偏好读取失败，无法可靠核对买家硬约束，请稍后重试") from error

    initial = await read()
    ranked = await selector.select(initial, query=query, top_k=top_k) if initial else []
    # 异步召回期间允许页面修改；返回时以一次权威读取统一提示与执行事实。
    facts = await read()
    selected = [p for p in facts if p.kind == "dislike"]
    selected.extend(p for candidate in ranked for p in facts
                    if p.kind == "like" and p == candidate and p not in selected)
    return ResolvedPreferences(facts, tuple(selected))
