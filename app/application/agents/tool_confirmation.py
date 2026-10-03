"""需要买家逐项批准的记忆写工具；执行策略在原生 MemoryApprovalMiddleware。"""

MEMORY_WRITE_TOOLS = frozenset({
    "remember_preference_tool", "update_preference_tool", "forget_preference_tool"
})
