"""执行关闭和新进程恢复共用持久收尾；不调用模型或重跑业务节点。"""
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Overwrite
from uuid import uuid4
from app.application.runtime.tool_history import tool_call_records
from app.application.runtime.finalization import RunPersistenceError


async def close_graph(session, reason, text):
    # END 先归并已完成兄弟节点的 pending writes；所有写入仍通过既有 fencing。
    await session.graph.aupdate_state(session.config, None, as_node=END)
    snapshot = await session.graph.aget_state(session.config)
    messages = list(snapshot.values.get('messages', []))
    missing, insertions = [], {}
    for record in tool_call_records(messages):
        if record.receipt is not None:
            continue
        receipt = ToolMessage(id=str(uuid4()), tool_call_id=record.call['id'], name=record.call['name'], status='error',
            content='本次调用未取得完整结果；涉及写入请先核查状态，不得直接重放。',
            artifact={'stop_reason': reason, 'execution_id': record.identifier})
        # 历史故障可能已跨过后续消息；回执仍紧跟其所属调用批次，不能挂到另一轮末尾。
        position = record.index + 1
        while position < len(messages) and isinstance(messages[position], ToolMessage):
            position += 1
        insertions.setdefault(position, []).append(receipt)
        missing.append(receipt)
    repaired = []
    for index in range(len(messages) + 1):
        repaired.extend(insertions.get(index, []))
        if index < len(messages):
            repaired.append(messages[index])
    repaired.append(AIMessage(id=str(uuid4()), content=text, additional_kwargs={'execution_stop': reason}))
    await session.graph.aupdate_state(session.config, {'messages': Overwrite(repaired)}, as_node='model')
    await session.graph.aupdate_state(session.config, None, as_node=END)
    return snapshot, missing


async def recover_graph(session):
    snapshot = await session.graph.aget_state(session.config)
    # 正常原生审批仍由用户决议恢复；它不是被遗弃的失败任务。
    if snapshot.interrupts:
        return snapshot
    messages = snapshot.values.get('messages', [])
    if snapshot.next or any(r.receipt is None for r in tool_call_records(messages)):
        try:
            await close_graph(session, 'server_restart',
                '上轮执行中断；已保存结果保留，未完成调用的结果尚未确认，不自动重放。')
        except Exception as error:
            raise RunPersistenceError('上轮图状态恢复未保存，不能继续执行') from error
        snapshot = await session.graph.aget_state(session.config)
    return snapshot
