"""BGE-M3 / BGE-reranker-v2-m3 专用 GPU 服务，无聊天或业务写入接口。"""
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import math
import logging
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

EMBEDDING_MODEL='BAAI/bge-m3'
RERANKER_MODEL='BAAI/bge-reranker-v2-m3'
Text=Annotated[str,Field(min_length=1,max_length=32768)]


class EmbeddingRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    model: str
    input: list[Text]=Field(min_length=1,max_length=32)
    encoding_format: Literal['float']='float'
    dimensions: Literal[1024] | None=None


class RerankRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    model: str
    query: str=Field(min_length=1,max_length=4096)
    documents: list[Text]=Field(min_length=1,max_length=128)
    top_n: int | None=Field(default=None,ge=1,le=128)


class Engine:
    def __init__(self, root, device):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        self.torch=torch;self.device=device
        self.revisions={name:json.loads((root/'models'/name/'globex-revision.json').read_text())['revision']
            for name in ('bge-m3','bge-reranker-v2-m3')}
        if not torch.cuda.is_available():raise RuntimeError('CUDA 不可用，未降级到 CPU')
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.35,device=device)
        self.embedding=SentenceTransformer(str(root/'models/bge-m3'),device=device,
            local_files_only=True,trust_remote_code=False,model_kwargs={'dtype':torch.float16})
        self.embedding.eval()
        self.tokenizer=AutoTokenizer.from_pretrained(str(root/'models/bge-reranker-v2-m3'),local_files_only=True,trust_remote_code=False)
        self.reranker=AutoModelForSequenceClassification.from_pretrained(str(root/'models/bge-reranker-v2-m3'),
            local_files_only=True,trust_remote_code=False,dtype=torch.float16).to(device).eval()
        self.embedding_limit=min(8192,self.embedding.max_seq_length)
        self.rerank_limit=min(8192,self.reranker.config.max_position_embeddings-2)
        if self.embedding.get_embedding_dimension()!=1024:raise RuntimeError('embedding 维度不匹配')

    @staticmethod
    def check_lengths(lengths,limit):
        if max(lengths)>limit:raise HTTPException(413,'单条输入超过模型窗口，未执行截断')
        if sum(lengths)>32768:raise HTTPException(413,'批量 token 过多，请分批请求')

    def embed(self,texts):
        lengths=[len(ids) for ids in self.embedding.tokenizer(texts,padding=False,truncation=False)['input_ids']]
        self.check_lengths(lengths,self.embedding_limit)
        with self.torch.inference_mode():
            vectors=self.embedding.encode(texts,batch_size=4,normalize_embeddings=True,
                show_progress_bar=False,convert_to_tensor=True).float().cpu().tolist()
        return vectors,sum(lengths)

    def rerank(self,query,documents):
        pairs=[[query,text] for text in documents]
        lengths=[len(ids) for ids in self.tokenizer(pairs,padding=False,truncation=False)['input_ids']]
        self.check_lengths(lengths,self.rerank_limit)
        scores=[]
        with self.torch.inference_mode():
            for start in range(0,len(pairs),4):
                inputs=self.tokenizer(pairs[start:start+4],padding=True,truncation=False,return_tensors='pt').to(self.device)
                scores.extend(self.reranker(**inputs).logits.reshape(-1).float().cpu().tolist())
        return scores,sum(lengths)


def create_app(root,api_key,device='cuda:0',engine_factory=Engine):
    if not api_key or len(api_key)<24:raise ValueError('必须设置独立服务访问密钥')
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine=await asyncio.to_thread(engine_factory,root,device)
        await asyncio.to_thread(app.state.engine.embed,['启动验证'])
        await asyncio.to_thread(app.state.engine.rerank,'背包',['旅行背包','咖啡机'])
        app.state.lock=asyncio.Lock();app.state.pending=0
        yield
    app=FastAPI(title='Globex BGE retrieval',lifespan=lifespan,docs_url=None,redoc_url=None)

    async def authorize(authorization: str | None=Header(default=None)):
        if not hmac.compare_digest((authorization or '').encode(),f'Bearer {api_key}'.encode()):
            raise HTTPException(401,'需要有效服务密钥')

    async def inference(function,*args):
        if app.state.pending>=8:raise HTTPException(429,'推理队列已满，请稍后重试')
        app.state.pending+=1
        try:
            async with app.state.lock:
                task=asyncio.create_task(asyncio.to_thread(function,*args))
                try:return await asyncio.shield(task)
                except asyncio.CancelledError:
                    # GPU 线程不能因 HTTP 取消被遗留并与下一请求并行。
                    try:await task
                    except Exception as error:logging.warning('取消后的推理结束异常：%s',type(error).__name__)
                    raise
        except HTTPException:raise
        except Exception as error:
            logging.error('模型推理失败：%s',type(error).__name__)
            raise HTTPException(503,'模型推理暂不可用，请稍后重试') from None
        finally:app.state.pending-=1

    @app.get('/health')
    async def health():
        return {'status':'ok','embedding':EMBEDDING_MODEL,'reranker':RERANKER_MODEL,
            'dimension':1024,'device':device,'embedding_max_tokens':app.state.engine.embedding_limit,
            'reranker_max_tokens':app.state.engine.rerank_limit,'pending':app.state.pending,'chat_model':False,
            'revisions':getattr(app.state.engine,'revisions',{})}

    @app.post('/v1/embeddings',dependencies=[Depends(authorize)])
    async def embeddings(request:EmbeddingRequest):
        if request.model!=EMBEDDING_MODEL:raise HTTPException(400,'模型名不匹配')
        vectors,tokens=await inference(app.state.engine.embed,request.input)
        if len(vectors)!=len(request.input) or any(len(v)!=1024 or not all(math.isfinite(x) for x in v) for v in vectors):
            raise HTTPException(502,'模型返回无效向量')
        return {'object':'list','model':EMBEDDING_MODEL,
            'data':[{'object':'embedding','index':i,'embedding':v} for i,v in enumerate(vectors)],
            'usage':{'prompt_tokens':tokens,'total_tokens':tokens}}

    @app.post('/v1/rerank',dependencies=[Depends(authorize)])
    async def rerank(request:RerankRequest):
        if request.model!=RERANKER_MODEL:raise HTTPException(400,'模型名不匹配')
        if request.top_n and request.top_n>len(request.documents):raise HTTPException(400,'top_n 超过候选数量')
        scores,tokens=await inference(app.state.engine.rerank,request.query,request.documents)
        if len(scores)!=len(request.documents) or not all(math.isfinite(x) for x in scores):
            raise HTTPException(502,'模型返回无效分数')
        results=sorted([{'index':i,'relevance_score':score} for i,score in enumerate(scores)],key=lambda row:row['relevance_score'],reverse=True)
        return {'model':RERANKER_MODEL,'results':results[:request.top_n or len(results)],
            'usage':{'input_tokens':tokens},'score_space':'logit'}
    return app


if __name__=='__main__':
    import uvicorn
    root=Path(os.environ.get('GLOBEX_RETRIEVAL_ROOT','/data4/sybai/globex'))
    api_key=(root/'service/api-key').read_text().strip()
    app=create_app(root,api_key,os.environ.get('BGE_DEVICE','cuda:0'))
    uvicorn.run(app,host='127.0.0.1',port=int(os.environ.get('BGE_PORT','18780')),access_log=False)
