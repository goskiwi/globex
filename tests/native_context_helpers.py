"""原生状态/消息夹具；策略直接执行，不模拟旧 Agent 类。"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from dataclasses import replace
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from langgraph.graph.message import add_messages
from app.application.runtime.context import RequestContextMiddleware
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from tests.test_retrieval import _settings


def history(count=8):
    messages=[]
    for i in range(count):
        messages.extend([HumanMessage(name='b',content=f'第{i+1}次搜索背包',id=f'u{i}'),
            AIMessage(id=f'a{i}',content='',tool_calls=[{'id':f'call{i}','name':'product_search_tool','args':{}}]),
            ToolMessage(id=f't{i}',tool_call_id=f'call{i}',name='product_search_tool',content=json.dumps({
                'hits':[{'product_id':f'P{i}','description':'商品限制'*100,'skus':[{'sku_id':f'P{i}-S1','stock':4}]}]},ensure_ascii=False), artifact={"data":{
                'hits':[{'product_id':f'P{i}','description':'商品限制'*100,'skus':[{'sku_id':f'P{i}-S1','stock':4}]}]}})])
    return {'messages':messages,'read_tool_messages':[f't{i}' for i in range(count)]}


def summary_selection(*sources):
    return json.dumps({'goals': list(sources or (0,)), 'decisions': [], 'open_questions': [], 'next_steps': []})


def policy(tmp_path,*,product_tokens=6000,target_tokens=48000,response=None,
           system_prompt="",tools=(),working_state_mode=None,**changes):
    model=SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=summary_selection() if response is None else response)))
    return RequestContextMiddleware(ContextEvidenceStore(tmp_path/'e.db'),model,
        replace(_settings(tmp_path),context_size=128000,context_product_tokens=product_tokens,
                context_target_tokens=target_tokens,**changes),system_prompt=system_prompt,tools=tools,
                working_state_mode=working_state_mode)


def apply(state,updates):
    return {**state,**updates,'messages':add_messages(state['messages'],updates.get('messages',[]))}
