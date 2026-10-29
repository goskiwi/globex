"""主动记忆读取：检索只返回候选，展示读取当前权威记录。"""
from dataclasses import asdict
from typing import Annotated, Literal
from pydantic import Field

from app.application.runtime.request_scope import ShoppingContext
from app.application.runtime.results import ToolResult,ToolResultState
from app.infrastructure.semantic_memory import MemoryUnavailable


def build_read_preferences_tool(store):
    async def read_preferences_tool(memory_ids: list[str] | None = None,
                                    kind: Literal['all','like','dislike'] = 'all') -> ToolResult:
        """查看当前买家已保存的长期偏好，返回真实ID、版本、原文依据和来源。

        memory_ids省略或null：列出kind范围内全部记忆，不按相关性截断。
        memory_ids指定：展示这些真实ID对应的当前记忆，不能猜测ID。
        定位相关记忆先用search_preferences_tool，再根据条目原文选择要展示的ID。
        查看全部、定位要修改/撤回的记忆时使用；修改仍需独立审批，不猜ID或版本。
        纯查看用turn_action=finish直接交付完整结果，之后还要修改或购物用continue。
        """
        context=ShoppingContext.current()
        if context is None:
            return ToolResult('缺少当前买家身份',state=ToolResultState.ERROR,
                              error_code='business_rejected',error_reason='缺少当前买家身份')
        try:
            facts=await store.list_by_buyer(context.buyer_id)
            if any(p.buyer_id!=context.buyer_id for p in facts):raise RuntimeError('偏好归属不一致')
            facts=[p for p in facts if kind=='all' or p.kind==kind]
            total=len(facts);scope='all' if memory_ids is None else 'selected'
            if memory_ids is not None:
                by_id={p.memory_id:p for p in facts}
                if any(identifier not in by_id for identifier in memory_ids):
                    raise ValueError('部分记忆已不存在或不属于当前读取范围，请重新查询。')
                facts=[by_id[identifier] for identifier in dict.fromkeys(memory_ids)]
        except ValueError as error:
            return ToolResult(str(error),state=ToolResultState.ERROR,error_code='business_rejected',error_reason=str(error))
        items=[{k:v for k,v in asdict(p).items() if k!='buyer_id'} for p in facts]
        label={'all':'长期偏好','like':'喜好','dislike':'禁忌'}[kind]
        if not total:
            guidance=f'当前没有已保存的{label}。'
        elif scope=='all':guidance=f'当前共保存{total}条{label}，以下为完整列表：'
        elif not items:guidance=f'本次未选择展示条目；当前仍保存{total}条{label}，可以列出全部核对。'
        else:guidance=f'当前共保存{total}条{label}，以下是本次选中的{len(items)}条：'
        lines=[f"{i}. {'喜好' if p.kind=='like' else '禁忌'}：{p.statement}" for i,p in enumerate(facts,1)]
        if lines:guidance+='\n'+'\n'.join(lines)
        return ToolResult({'operation':'memory','action':'read','status':'completed',
            'scope':scope,'kind':kind,'total_count':total,
            'returned_count':len(items),'preferences':items,'guidance':guidance})
    return read_preferences_tool


def build_search_preferences_tool(store):
    async def search_preferences_tool(query: str, kind: Literal['all','like','dislike'] = 'all',
                                     limit: Annotated[int,Field(ge=1,le=200,strict=True)] = 5) -> ToolResult:
        """检索当前买家的记忆候选，包含真实ID、版本、原文依据及相似度。

        相关性排序不是已确认的语义匹配。依据用户问题和条目原文选择相关项，再用
        read_preferences_tool(memory_ids=...)展示，或用当前ID/版本申请修改、撤回。
        查询失败不等于没有记忆；无候选时结合total_count说明范围。读取不授予写入批准。
        """
        context=ShoppingContext.current()
        if context is None:return ToolResult('缺少当前买家身份',state=ToolResultState.ERROR,
            error_code='business_rejected',error_reason='缺少当前买家身份')
        try:
            result=await store.search(context.buyer_id,query,limit=limit,kind=None if kind=='all' else kind)
        except (MemoryUnavailable,NotImplementedError):
            message='相关记忆查询暂不可用，不能据此判断没有保存；可以列出全部记忆核对。'
            return ToolResult({'status':'unavailable','query':query,'guidance':message},
                state=ToolResultState.ERROR,error_code='unavailable',error_reason=message)
        except ValueError as error:
            return ToolResult(str(error),state=ToolResultState.ERROR,error_code='business_rejected',error_reason=str(error))
        if any(m.preference.buyer_id!=context.buyer_id for m in result.matches):raise RuntimeError('偏好归属不一致')
        candidates=[{**{k:v for k,v in asdict(m.preference).items() if k!='buyer_id'},'score':m.score}
                    for m in result.matches]
        return ToolResult({'status':'completed','query':query,'kind':kind,'total_count':result.total_count,
            'candidate_count':result.matched_count,'returned_count':len(candidates),
            'has_more':len(candidates)<result.matched_count,'candidates':candidates})
    return search_preferences_tool
