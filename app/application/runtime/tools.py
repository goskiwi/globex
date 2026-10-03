# -*- coding: utf-8 -*-
"""把现有业务函数适配为 LangChain StructuredTool。"""
from __future__ import annotations

import functools
import inspect
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from typing import Any

from langchain_core.tools import StructuredTool
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from app.application.runtime.results import ToolResult, input_failure
from pydantic import BaseModel, ConfigDict, ValidationError, create_model


def validation_feedback(error):
    return "参数校验失败：\n" + "\n".join(
        f"{'.'.join(map(str, item['loc'])) or '参数组合'}: {item['msg']}"
        for item in error.errors(include_input=False, include_context=False, include_url=False))


class ToolInputError(Exception):
    """仅代表原生输入解析失败，不包含工具业务体抛出的校验异常。"""
    def __init__(self, error, schema):
        self.failure = input_failure(error,schema)
        self.feedback = validation_feedback(error)
        super().__init__('tool input validation failed')


class ToolExecutionError(RuntimeError):
    """工具体异常与输入错误分开，防止 ToolNode 把业务体校验异常当参数错误。"""


class TypedTool(StructuredTool):
    """保留输入模型的约束配置；原生子集 Schema 会丢失 extra/json_schema_extra。"""
    handle_validation_error: bool | str | Callable = False

    def _parse_input(self, tool_input, tool_call_id):
        try:
            return super()._parse_input(tool_input, tool_call_id)
        except ValidationError as error:
            schema=self.tool_call_schema
            raise ToolInputError(error,schema.model_json_schema() if isinstance(schema,type) else schema) from error

    async def ainvoke(self, input, config=None, **kwargs):
        try:
            return await super().ainvoke(input, config, **kwargs)
        except ToolInputError as error:
            # 由本次真实校验产生一次失败回执；不预校验、不解析反馈文字。
            if not isinstance(input, dict) or input.get('type') != 'tool_call':
                raise
            return ToolMessage(content=error.feedback, name=self.name, tool_call_id=input['id'],
                status='error', artifact={'failure': error.failure, 'error_code': 'invalid_input'})
        except ValidationError as error:
            raise ToolExecutionError('工具体校验异常') from error
    @property
    def tool_call_schema(self):
        visible = super().tool_call_schema
        if isinstance(visible, type) and issubclass(visible, BaseModel) and isinstance(self.args_schema, type):
            return create_model(visible.__name__, __config__=self.args_schema.model_config,
                **{name:(self.args_schema.model_fields[name].annotation, self.args_schema.model_fields[name])
                   for name in visible.model_fields})
        return visible


def _message(value, runtime, name):
    """错误类别走运行时元数据；模型看到业务解释，不靠文字前缀推断状态。"""
    if isinstance(value, ToolResult):
        artifact={"data":value.data, "notices":[],
                  "error_code":value.error_code.value if value.error_code else None}
        if value.error_reason is not None: artifact['error_reason'] = value.error_reason
        if value.model_data is not None: artifact["model_data"]=value.model_data
        if value.event_data is not None: artifact["event_data"]=value.event_data
        if name == "task_dispatch" and isinstance(value.data, dict) and value.data.get("stop_reason"):
            artifact["stop_reason"] = value.data["stop_reason"]
        return ToolMessage(content=value.text, tool_call_id=runtime.tool_call_id, name=name,
            status=value.state.value, artifact=artifact)
    import json
    return ToolMessage(content=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False),
        tool_call_id=runtime.tool_call_id, name=name, artifact={"data":value,"notices":[]})


async def _collect_async_generator(generator: AsyncGenerator[Any, None], runtime, name):
    chunks: list[str] = []
    async with aclosing(generator):
        async for value in generator:
            if isinstance(value, ToolResult) and not value.ok:
                return _message(value, runtime, name)
            chunks.append(value.text if isinstance(value,ToolResult) else str(value))
    return "\n".join(part for part in chunks if part)


def as_langchain_tool(function: Callable[..., Any], *, name: str | None = None, args_schema=None) -> StructuredTool:
    """保留原函数签名和文档，统一异步函数及异步生成器的输出。"""
    signature = inspect.signature(function)
    args_schema = args_schema or getattr(function, "input_model", None)
    runtime_name = "runtime" if "runtime" in signature.parameters else "tool_runtime"

    @functools.wraps(function)
    async def invoke(**kwargs: Any) -> str:
        runtime = kwargs[runtime_name]
        if runtime_name == "tool_runtime":
            kwargs.pop(runtime_name)
        result = function(**kwargs)
        if inspect.isasyncgen(result):
            return await _collect_async_generator(result, runtime, name or function.__name__)
        if inspect.isawaitable(result):
            result = await result
        return _message(result, runtime, name or function.__name__)

    invoke.__signature__ = signature if runtime_name == "runtime" else signature.replace(parameters=[
        *signature.parameters.values(), inspect.Parameter(runtime_name, inspect.Parameter.KEYWORD_ONLY, annotation=ToolRuntime)])
    invoke.__annotations__ = {**function.__annotations__, runtime_name: ToolRuntime}
    if args_schema is not None:
        args_schema = create_model(args_schema.__name__ + "Runtime", __base__=args_schema,
            __config__=ConfigDict(arbitrary_types_allowed=True), **{runtime_name: (ToolRuntime, ...)})
    tool = TypedTool.from_function(
        coroutine=invoke,
        name=name or function.__name__,
        description=inspect.getdoc(function) or function.__name__,
        args_schema=args_schema,
    )
    tool.args_schema = create_model(tool.args_schema.__name__ + "Input", __base__=tool.args_schema,
                                   __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
    return tool
