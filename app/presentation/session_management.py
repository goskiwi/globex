"""签名用户的对话重命名与删除，操作不进入Agent工具循环。"""
from fastapi import FastAPI,Request,HTTPException,Query
from pydantic import BaseModel,ConfigDict,Field
from app.presentation.identity import require_buyer,session_error
from app.infrastructure.ag_ui_journal import JournalForbidden,JournalNotFound,JournalConflict
from app.domain.session.ports.session_store import SessionStoreError
from app.infrastructure.queue.redis_stream_queue import SessionLeaseTimeout


class RenameRequest(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True,str_strip_whitespace=True)
    title:str=Field(min_length=1,max_length=100)


def register_session_management_routes(api:FastAPI,get_service):
    def translate(error):
        if isinstance(error,SessionStoreError):return session_error(error)
        code=403 if isinstance(error,JournalForbidden) else 404 if isinstance(error,JournalNotFound) else 409
        return HTTPException(code,str(error))

    @api.patch('/commerce/ag-ui/sessions/{session_id}')
    async def rename(session_id:str,body:RenameRequest,request:Request,buyer_id:str=Query(min_length=1)):
        await require_buyer(request,buyer_id)
        try:return await get_service().rename(buyer_id,session_id,body.title)
        except (SessionStoreError,JournalForbidden,JournalNotFound,JournalConflict,SessionLeaseTimeout) as error:
            raise translate(error) from error

    @api.delete('/commerce/ag-ui/sessions/{session_id}')
    async def delete(session_id:str,request:Request,buyer_id:str=Query(min_length=1)):
        await require_buyer(request,buyer_id)
        try:return await get_service().delete(buyer_id,session_id)
        except (SessionStoreError,JournalForbidden,JournalNotFound,JournalConflict,SessionLeaseTimeout) as error:
            raise translate(error) from error
