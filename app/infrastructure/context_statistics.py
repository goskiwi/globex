"""上下文装配与评测共用的统计定义；估算值不等于供应商 usage。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

TOKEN_SECTIONS = ('system_tokens', 'tool_tokens', 'skill_tokens', 'state_tokens',
                  'other_fixed_tokens', 'history_tokens', 'protocol_tokens')


class RequestParts(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    system_tokens: int = Field(ge=0)
    tool_tokens: int = Field(ge=0)
    skill_tokens: int = Field(ge=0)
    state_tokens: int = Field(ge=0)
    other_fixed_tokens: int = Field(ge=0)
    history_tokens: int = Field(ge=0)
    protocol_tokens: int = Field(ge=0)
    fixed_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    input_limit: int
    history_budget: int = Field(ge=0)


class SummaryAttempt(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    attempt: int = Field(ge=1)
    status: Literal['accepted', 'rejected']


class ContextStatistics(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    status: Literal['completed', 'noop', 'failed']
    before_tokens: int = Field(ge=0)
    after_tokens: int = Field(ge=0)
    archived_result_count: int = Field(ge=0)
    summary_attempts: list[SummaryAttempt]
    request_parts_before: RequestParts
    request_parts_after: RequestParts
