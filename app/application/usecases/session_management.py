"""对话管理：复用执行租约与归属，清理内容，不触碰独立业务资产。"""
class SessionManagement:
    def __init__(self, orchestrator, journal, session_store, conversations, evidence, forms, checkpointer):
        self.orchestrator,self.journal,self.session_store=orchestrator,journal,session_store
        self.conversations,self.evidence,self.forms,self.checkpointer=conversations,evidence,forms,checkpointer

    async def rename(self,buyer_id,session_id,title):
        await self.session_store.assert_owner(session_id,buyer_id,create=False)
        return await self.journal.rename_session(session_id,buyer_id,title)

    async def delete(self,buyer_id,session_id):
        await self.session_store.assert_owner(session_id,buyer_id,create=False)
        await self.journal.require_idle(session_id,buyer_id)
        async with self.orchestrator.session_operation(session_id):
            async def cleanup():
                # 先终止旧执行权，即使跨库清理失败也不能让旧任务复活；可重试同一清理。
                await self.session_store.retire_conversation(session_id,buyer_id)
                await self.orchestrator._sessions.invalidate(session_id)
                await self.checkpointer.adelete_thread(session_id)
                await self.evidence.delete_session(buyer_id,session_id)
                await self.forms.delete_session(buyer_id,session_id)
                await self.conversations.delete_conversation(session_id,buyer_id)
            return await self.journal.delete_session(session_id,buyer_id,cleanup)
