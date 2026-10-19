"""原生工具边界：上下文类型正确、注入对象身份保留、业务参数依旧严格校验。"""
import warnings

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage,ToolMessage
from pydantic import BaseModel,ConfigDict,Field

from app.application.runtime.handoff import HandoffContext,TaskEvidence
from app.application.runtime.results import ToolResult
from app.application.runtime.tools import as_langchain_tool
from tests.native_tool_helpers import SingleToolModel


class BusinessInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    count:int=Field(gt=0)


@pytest.mark.parametrize('explicit_schema',[False,True])
@pytest.mark.parametrize('own_runtime',[False,True])
@pytest.mark.parametrize('with_context',[False,True])
async def test_native_runtime_context_retains_identity_without_serialization_warning(explicit_schema,own_runtime,with_context):
    observed=[]
    async def ordinary(count:int):
        """读取业务数量。"""
        observed.append(count)
        return ToolResult(data={'count':count})
    async def contextual(count:int,runtime:ToolRuntime):
        """读取服务端上下文，不把它当作模型参数。"""
        observed.append(runtime.context)
        return ToolResult(data={'count':count})
    target=contextual if own_runtime else ordinary
    tool=as_langchain_tool(target,args_schema=BusinessInput if explicit_schema else None)
    context=HandoffContext(TaskEvidence('opaque-context')) if with_context else None
    if context:context.evidence.refs.add('server-only-evidence')
    graph=create_agent(SingleToolModel(selected_tool=tool.name),tools=[tool],context_schema=HandoffContext if with_context else None)
    # 不屏蔽任何警告；从原生工具解析直至执行都按严格模式验证。
    with warnings.catch_warnings():
        warnings.simplefilter('error',UserWarning)
        result=await graph.ainvoke({'messages':[HumanMessage(content='{"count":2}')]},context=context)
    receipt=next(m for m in result['messages'] if isinstance(m,ToolMessage))
    assert receipt.status=='success' and receipt.artifact['data']=={'count':2}
    assert observed==([context] if own_runtime else [2])
    if own_runtime:assert observed[0] is context
    assert set(tool.tool_call_schema.model_fields)=={'count'}
    runtime_name='runtime' if own_runtime else 'tool_runtime'
    runtime=ToolRuntime(state={},context=context,config={},stream_writer=lambda _:None,tool_call_id='call',store=None)
    with warnings.catch_warnings():
        warnings.simplefilter('error',UserWarning)
        validated=tool.args_schema.model_validate({'count':2,runtime_name:runtime})
        dumped=validated.model_dump()
        assert dumped['count']==2 and runtime_name in dumped
    assert getattr(validated,runtime_name) is runtime
    if context:assert runtime.context.evidence is context.evidence


@pytest.mark.parametrize('explicit_schema',[False,True])
@pytest.mark.parametrize('payload',['{"count":"invalid"}','{"count":2,"undeclared":true}'])
async def test_context_type_does_not_weaken_business_argument_validation(explicit_schema,payload):
    observed=[]
    async def execute(count:int):
        """只允许已声明的业务参数。"""
        observed.append(count)
        return ToolResult(data={'count':count})
    tool=as_langchain_tool(execute,args_schema=BusinessInput if explicit_schema else None)
    graph=create_agent(SingleToolModel(selected_tool=tool.name),tools=[tool],context_schema=HandoffContext)
    with warnings.catch_warnings():
        warnings.simplefilter('error',UserWarning)
        result=await graph.ainvoke({'messages':[HumanMessage(content=payload)]},context=HandoffContext(TaskEvidence('invalid-business')))
    receipt=next(m for m in result['messages'] if isinstance(m,ToolMessage))
    assert receipt.status=='error' and receipt.artifact['error_code']=='invalid_input'
    assert observed==[]
