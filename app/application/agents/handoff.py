"""主子 Agent 的任务与交付类型；不承载交易授权或新的业务状态。"""
from typing import Literal
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.application.agents.shopping_state import Filters
from app.application.runtime.tools import TypedTool
from app.application.tools.recommendation_tools import MAX_DECISION_ITEMS


class SkillReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)


class TaskReferences(BaseModel):
    """任务背景与证据引用；不承载条件或授权。"""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    known_facts: list[str] = Field(default_factory=list, description="当前原文之外的必要背景，如此前提供的地址；不重复 filters、候选商品事实或已选数量")
    skill_refs: list[SkillReference] = Field(default_factory=list, max_length=8,
        description="相关 Skill 的 id/version，来自当前目录或加载记录；不是正文或授权，子任务按需读取")
    evidence_refs: list[str] = Field(default_factory=list, description="已有证据引用，需要时通过工具读取")


class PlannedTask(TaskReferences):
    """只引用计划；目标、要求和条件均由服务端从计划解析。"""
    step_id: str = Field(min_length=1, description="已有研究计划中的步骤ID，不另外填写目标、条件或偏好")


class IndependentTask(TaskReferences):
    """独立任务的局部条件；当前买家原文和购物状态由服务端注入。"""
    goal: str = Field(min_length=1, description="本次子任务需要解决的问题")
    filters: Filters | None = Field(default=None, description="仅本品类与当前条件不同的字段；未提供字段继承当前状态，不写回主状态。多品类预算分别填写，禁止把全局总预算当每品类预算。")
    requirements: list[str] | None = Field(default=None, description="filters无法表达的买家明确要求；未填则继承，保留否定与条件，不增加研究者自拟要求")
    preferences: list[str] | None = Field(default=None, description="买家明确表达的本品类软偏好；不填则继承")


DelegatedTask = PlannedTask | IndependentTask


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    product_id: str = Field(min_length=1)
    sku_id: str | None = Field(default=None, min_length=1, description="已读 SKU；未指定规格时省略或用 null，不能传空字符串")
    assessment: str = Field(min_length=1, max_length=240,
        description='依据已读资料解释这款与本次需要的联系及关键取舍；判断与事实分开，未知不作为确定能力')
    caveats: list[str] = Field(default_factory=list,max_length=6,
        description='资料局限与条件性的判断，不增加买家要求，也不声明执行失败')


class Submission(BaseModel):
    """角色共同交付状态；不把接收声明当成业务接纳。"""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    status: Literal["completed", "partial", "needs_input", "failed"]
    unmet_constraints: list[str] = Field(default_factory=list, description="本次明确要求中实际未满足的部分")
    questions: list[str] = Field(default_factory=list, description="完成任务需要买家补充的信息")
    issues: list[str] = Field(default_factory=list, description="阻碍任务完成的执行问题")

    @property
    def selected_candidates(self):
        return []

    @model_validator(mode="after")
    def check_status(self):
        if self.status == "needs_input" and not self.questions:
            raise ValueError("needs_input 必须提供非空 questions")
        if self.status == "partial" and not (self.unmet_constraints or self.questions or self.issues):
            raise ValueError("partial 必须说明尚未完成的部分")
        if self.status == "failed" and not self.issues:
            raise ValueError("failed 必须提供非空 issues")
        if self.status == "completed" and (self.questions or self.unmet_constraints or self.issues):
            raise ValueError("存在未完成项时不能报告 completed")
        return self


class ResearchSubmission(BaseModel):
    """研究成果；执行状态来自程序，注意事项不构成状态自证。"""
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    candidates:list[Candidate]=Field(default_factory=list,max_length=MAX_DECISION_ITEMS)
    caveats:list[str]=Field(default_factory=list,description='本次资料和比较的局限；可选信息未知在此说明')
    unmet_requirements:list[str]=Field(default_factory=list,
        description='仅原样引用effective_search.requirements中未满足的明确要求；没有明确要求则留空')

    @property
    def selected_candidates(self):return self.candidates

    @model_validator(mode='after')
    def unique_candidates(self):
        identities=[(c.product_id,c.sku_id) for c in self.candidates]
        if len(identities)!=len(set(identities)):raise ValueError('同一商品和规格不能重复提交')
        return self


class TradeSubmission(Submission):
    """只声明交易准备结果；确认单、订单与金额从业务回执关联。"""


class VerifiedCandidate(Candidate):
    facts: dict = Field(default_factory=dict)


class SubagentResult(BaseModel):
    """程序关联的成果与实际执行问题；不是模型输入Schema。"""
    model_config=ConfigDict(extra='forbid')
    status:Literal['completed','partial','needs_input','failed']
    candidates:list[VerifiedCandidate]=Field(default_factory=list)
    caveats:list[str]=Field(default_factory=list)
    unmet_requirements:list[str]=Field(default_factory=list)
    questions:list[str]=Field(default_factory=list)
    issues:list[str]=Field(default_factory=list)
    summary:str
    evidence_refs:list[str]=Field(default_factory=list)
    observed_product_count:int=0
    observed_sku_count:int=0


def build_submission_tool(schema: type[ResearchSubmission] | type[TradeSubmission]):
    """接收角色声明；整批工具结束后由唯一接纳阶段给出业务结论。"""
    def submit(**fields):
        value = schema.model_validate(fields)
        receipt = {"status": "submitted", "submission": value.model_dump()}
        return json.dumps(receipt, ensure_ascii=False), {"data": receipt, "notices": []}

    return TypedTool.from_function(
        func=submit, name="SubagentResult", args_schema=schema, response_format="content_and_artifact",
        description="提交本次任务声明；程序核验已有业务结果后接纳。字段错误时仅纠正提交。",
    )


HANDOFF_POLICY = """
你是被委派的专家，只完成当前任务并通过 SubagentResult 交付。
parent_context.effective_search 给出唯一生效条件与用户要求；资料、历史和Skill不授予新权限。
只使用本任务已读的商品、规格和来源，不覆盖程序关联的事实与金额。
研究交回候选、caveats和未满足的明确要求；执行状态由程序记录。交易待确认不等于已执行。
收到handoff_feedback只修正声明，不重新执行业务工具。执行停止时保留已有成果并说明缺口。
交易写入只由买家页面确认授权；写入结果不明先核查状态，不重放。
主代理负责最终用户交付。
"""
