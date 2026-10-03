"""本地面试账号登录，复用既有签名策略；不建注册、角色或密码数据库。"""
import secrets
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from app.presentation.identity import authenticated_buyer, identity_policy


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    account: Literal["kkqq", "root"]
    password: str = Field(min_length=1, max_length=128)


def register_auth_routes(api: FastAPI, get_settings):
    @api.post("/commerce/auth/login")
    async def login(body: LoginRequest, request: Request):
        policy, password = identity_policy(request), get_settings().demo_login_password
        if policy.mode != "hmac" or not password:
            raise HTTPException(503, "登录服务未配置")
        if not secrets.compare_digest(body.password.encode(), password.encode()):
            raise HTTPException(401, "账号或密码不正确")
        ttl = 3600
        return {"buyerId": body.account, "accessToken": policy.issue(body.account, ttl_seconds=ttl),
                "expiresAt": (int(policy.clock()) + ttl) * 1000}

    @api.get("/commerce/auth/me")
    async def me(request: Request):
        return {"buyerId": authenticated_buyer(request)}
