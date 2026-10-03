"""工具验证使用真实 create_agent/StructuredTool，只隔离模型决策。"""
import json
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from app.application.runtime.tools import as_langchain_tool


class SingleToolModel(BaseChatModel):
    selected_tool: str

    @property
    def _llm_type(self):return 'native-tool-test'

    def bind_tools(self, tools, **kwargs):return self

    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        if any(isinstance(message,ToolMessage) for message in messages):
            result=AIMessage(content='完成')
        else:
            result=AIMessage(content='',tool_calls=[{'id':'call','name':self.selected_tool,
                'args':json.loads(next(m.text for m in messages if isinstance(m,HumanMessage)))}])
        return ChatResult(generations=[ChatGeneration(message=result)])


def tool_graph(function, *, middlewares):
    tool=function if isinstance(function,BaseTool) else as_langchain_tool(function)
    return create_agent(SingleToolModel(selected_tool=tool.name),tools=[tool],middleware=middlewares)


async def call_tool(graph, **kwargs):
    result=await graph.ainvoke({'messages':[HumanMessage(content=json.dumps(kwargs))]})
    return next(message for message in reversed(result['messages']) if isinstance(message,ToolMessage))
