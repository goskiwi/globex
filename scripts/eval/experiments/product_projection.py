"""仅用于固定轨迹的说明去重实验，不参与正式请求装配。"""
from langchain_core.messages import ToolMessage
from app.application.runtime.projections import share_identical_products
from app.application.runtime.results import message_data, project_data


def compact_product_messages(messages):
    projected = share_identical_products(messages)
    for index, message in enumerate(projected):
        if not isinstance(message, ToolMessage):
            continue
        payload = message_data(message)
        if not isinstance(payload, dict):
            continue
        payload.pop('shared_fields_notice', None)
        if payload.get('archived'):
            payload.pop('notice', None)
        projected[index] = project_data(message, payload)
    return projected
