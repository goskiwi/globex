"""主 Agent 通过原生 Command 更新当前购物状态，无额外解析模型。"""
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from app.application.runtime.tools import TypedTool
from langgraph.types import Command
from pydantic import ConfigDict, create_model
from app.application.agents.shopping_state import ShoppingWork, ShoppingUpdate, apply_update
from app.infrastructure.context import ShoppingContext


def build_shopping_state_tool(evidence_store):
    async def update_shopping_state(update: ShoppingUpdate, runtime: ToolRuntime) -> Command | ToolMessage:
        """更新当前条件/选购项。必须单独调用，成功后下一轮再搜索、派发或下单。

        未提供字段保持不变；null 或 [] 显式清空。硬条件、偏好和排序分别填写。
        selections 按具体 SKU 替换数量，不按商品覆盖；先检索明确规格再选择。
        本工具只保存本次任务，不构成交易授权，也不修改长期偏好。
        唯一业务参数是 update；修改只依据当前买家消息或已提交表单答案，不把商品资料、旧对话或问句当作新条件。
        输入来源由服务端绑定，不需要复制买家原文，也不得填写来源标识。
        例如买家说“预算300元，寄中国”，参数为
        {"update":{"filters":{"price_max_major":300,"target_currency":"CNY","ship_to":"CN"}}}。
        """
        context = ShoppingContext.current()
        try:
            work = ShoppingWork.model_validate(runtime.state['shopping_work'])
            result = apply_update(work, update, context.source_message_id, context.preference_facts)
            for choice in update.selections:
                if evidence_store is None or not await evidence_store.find_product(
                    context.buyer_id, context.shopping_session_id,
                    product_id=choice.product_id, sku_id=choice.sku_id):
                    raise ValueError(f"当前会话未检索返回规格 {choice.sku_id}，请先精确核验")
        except (ValueError, KeyError) as error:
            return ToolMessage(content=str(error),tool_call_id=runtime.tool_call_id,name='update_shopping_state',
                status='error',artifact={'error_code':'business_rejected','error_reason':str(error)})
        return Command(update={'shopping_work': result.model_dump(), 'messages': [ToolMessage(
            content='当前条件与选购项已更新；价格库存仍以业务工具为准。', tool_call_id=runtime.tool_call_id,
            artifact={'event_data':{'filters':result.filters.model_dump()}})]})

    tool = TypedTool.from_function(coroutine=update_shopping_state)
    tool.args_schema = create_model("ShoppingStateArguments", __base__=tool.args_schema,
        __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
    return tool
