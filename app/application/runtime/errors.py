"""运行时共享异常及原生工具批次的停止信号。"""


class ContextCapacityError(ValueError):
    pass


class ExecutionStopped(RuntimeError):
    """执行限制已触发，不是模型提交错误，也不是可重试的服务故障。"""
    REASONS = {
        "repeated_path": "纠偏后仍重复相同路径且业务结果未变化，已结束本轮自主探索",
        "output_limit": "模型输出达到上限，未完整生成回答",
        "provider_refusal": "模型服务拒绝生成本次回答",
        "budget_exhausted": "本轮预算不足，已停止继续调用模型",
        "step_limit": "本轮已达到执行步数上限",
        "model_call_limit": "本轮已达到模型调用轮数上限",
        "timeout": "本次子任务已达到执行时间上限",
        "context_capacity": "本次任务输入超过安全上下文容量，已停止继续调用模型",
    }

    def __init__(self, reason: str):
        if reason not in self.REASONS:
            raise ValueError("未知停止原因")
        self.reason = reason
        super().__init__(self.REASONS[reason])


def raise_if_tool_stopped(messages):
    """只检查当前工具批次；运行时反馈消息不能掩盖已发生的停止。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            break
        if isinstance(message, HumanMessage) and message.name not in {
            "handoff_feedback", "shopping_state", "shopping_state_delta", "context_summary"
        }:
            break
        if isinstance(message, ToolMessage) and isinstance(message.artifact, dict):
            reason = message.artifact.get("stop_reason")
            if reason in ExecutionStopped.REASONS:
                raise ExecutionStopped(reason)
