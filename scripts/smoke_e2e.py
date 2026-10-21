# -*- coding: utf-8 -*-
"""端到端冒烟脚本

用法（先启动服务）：
    uv run uvicorn app.presentation.server:app --port 8000
    uv run python scripts/smoke_e2e.py [--query "..."]

流程：POST 持久 AG-UI 运行 → 读取公开 SSE 与最终回复。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid

import httpx
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_stream import submit_stream

BASE_URL = "http://127.0.0.1:8000"
BUYER_ID = os.getenv("GLOBEX_BUYER_ID", "buyer-001")
API_TOKEN = os.getenv("GLOBEX_API_TOKEN", "")
AUTH_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


def validate_smoke_result(body: dict, events: list[dict]) -> None:
    """冒烟必须同时拿到最终回复和完整工具事件，缺一即失败。"""
    if not str(body.get("final_text", "")).strip():
        raise AssertionError("冒烟失败：响应缺少非空 final_text")
    event_types = {event.get("type") for event in events}
    for required in ("TOOL_CALL_START", "TOOL_CALL_RESULT", "RUN_FINISHED"):
        if required not in event_types:
            raise AssertionError(f"冒烟失败：事件流缺少 {required}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="我想买一套便宜又抗造的旅行三件套，预算 300 块，最好不要塑料的，喜欢小众一点。",
    )
    parser.add_argument("--session", default=None, help="复用已有会话 ID 做多轮对话")
    args = parser.parse_args()

    session_id = args.session or f"smoke-{uuid.uuid4().hex[:8]}"
    print(f"== 会话：{session_id}\n== 意图：{args.query}\n")

    async with httpx.AsyncClient(timeout=600, headers=AUTH_HEADERS) as client:
        stream,_ = await submit_stream(client,BASE_URL,session_id,BUYER_ID,args.query)
        stream.require_success()
        body,events = {"final_text":stream.final_text},stream.frames
        for event in events:
            print(f"[{event['type']}] {json.dumps(event,ensure_ascii=False)[:300]}")
    validate_smoke_result(body, events)

    print("\n\n===== 最终回复 =====")
    print(body["final_text"])
    print("\n===== 事件统计 =====")
    counts: dict[str, int] = {}
    for event in events:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
