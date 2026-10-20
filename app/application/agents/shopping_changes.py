"""购物状态的领域增量；并行工具只提交自己实际改变的字段和规格。"""
from copy import deepcopy
from typing import Any

from pydantic import model_validator

from app.application.agents.shopping_state import StrictModel, ShoppingWork, Filters, Choice


class ShoppingChange(StrictModel):
    source_message_id: str
    fields: dict[str, Any]
    filters: dict[str, Any]
    selections: dict[str, Choice]
    remove_skus: list[str]
    reset: bool = False

    @model_validator(mode='after')
    def known_fields(self):
        if set(self.fields) - (set(ShoppingWork.model_fields) - {'filters','selections','source_message_id','latest_request'}):
            raise ValueError('购物增量包含未知字段或来源字段')
        if set(self.filters) - set(Filters.model_fields):
            raise ValueError('购物增量包含未知过滤字段')
        if any(key != value.sku_id for key,value in self.selections.items()):
            raise ValueError('购物增量规格主键不一致')
        return self

    @classmethod
    def between(cls, before: ShoppingWork, after: ShoppingWork, *, reset=False):
        old,new=before.model_dump(),after.model_dump()
        fields={key:value for key,value in new.items()
                if key not in {'filters','selections','source_message_id','latest_request'}
                and (reset or old[key]!=value)}
        return cls(source_message_id=after.source_message_id,fields=fields,
            filters={key:value for key,value in new['filters'].items() if reset or old['filters'][key]!=value},
            selections={key:value for key,value in after.selections.items()
                        if reset or before.selections.get(key)!=value},
            remove_skus=[key for key in before.selections if key not in after.selections],reset=reset)


def apply_shopping_change(current: dict | None, incoming: ShoppingChange | dict) -> dict:
    # 完整当前结构只用于图的初始输入/恢复。业务工具统一提交ShoppingChange，不猜测旧结构。
    if not isinstance(incoming,ShoppingChange):
        if current is not None:
            raise ValueError('完整购物状态只能用于初始化，业务工具必须提交领域增量')
        try:
            return ShoppingWork.model_validate(incoming).model_dump()
        except ValueError:
            raise ValueError('旧购物状态不再支持继续执行，请新建会话；历史记录保留') from None
    work=ShoppingWork.model_validate(current)
    if not incoming.source_message_id or incoming.source_message_id!=work.source_message_id:
        raise ValueError('购物增量没有绑定当前买家输入')
    if incoming.reset:
        work=ShoppingWork(filters=Filters(),source_message_id=work.source_message_id,latest_request=work.latest_request)
    data=work.model_dump()
    data.update(deepcopy(incoming.fields))
    data['filters'].update(incoming.filters)
    for sku in incoming.remove_skus:
        data['selections'].pop(sku,None)
    data['selections'].update({key:value.model_dump() for key,value in incoming.selections.items()})
    # 排除操作与已有选择共用领域不变量；不会把已排除规格留在状态里。
    data['selections']={key:value for key,value in data['selections'].items()
                        if key not in data['excluded_skus'] and value['product_id'] not in data['excluded_products']}
    return ShoppingWork.model_validate(data).model_dump()
