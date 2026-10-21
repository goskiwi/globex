# -*- coding: utf-8 -*-
"""真并行 fork 验证脚本

用法（先启动服务）：
    uv run uvicorn app.presentation.server:app --port 8000
    uv run python scripts/verify_parallel.py

原理：同一个"三品类分头调研"意图跑两遍——
    并行版：prompt 已要求同一轮一次性发起多个 task_dispatch（2.0 会并发批执行）
    串行版：在 query 里显式要求"一个一个来，做完一个再做下一个"
对比 wall time，并从 持久 SSE 业务事件的 agent.dispatch.started_at / tool.result.finished_at
判断多次派发的时间区间是否重叠（重叠即真并行）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from datetime import datetime

import httpx
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_stream import submit_stream

BASE_URL = "http://127.0.0.1:8000"
BUYER_ID = os.getenv("GLOBEX_BUYER_ID", "buyer-001")
API_TOKEN = os.getenv("GLOBEX_API_TOKEN", "")
AUTH_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

PARALLEL_QUERY = (
    "我下个月去高原露营，请分头调研三类装备：露营照明、登山杖、速干毛巾。"
    "三件事彼此独立，请一次性并行派三个检索子代理去做，最后汇总推荐。"
)
SERIAL_QUERY = (
    "我下个月去高原露营，请调研三类装备：露营照明、登山杖、速干毛巾。"
    "请严格一个一个来：先把露营照明彻底做完，再做登山杖，最后做速干毛巾，"
    "每次只派一个检索子代理，不要同时派多个。"
)


def _dispatch_intervals(events: list[dict]) -> list[tuple[str, str, str]]:
    """从事件流提取 (agent, started_at, finished_at) 三元组。"""
    intervals = []
    from scripts.eval.trace import tool_observations
    for result in tool_observations(events):
        payload = result['data']
        if result['name'] == 'task_dispatch' and result['status'] == 'success' and isinstance(payload, dict):
            if all(payload.get(field) for field in ('agent', 'started_at', 'finished_at')):
                intervals.append((payload['agent'], payload['started_at'], payload['finished_at']))
    return intervals


def _count_overlaps(intervals: list[tuple[str, str, str]]) -> int:
    parsed = [
        (datetime.fromisoformat(start), datetime.fromisoformat(end))
        for _, start, end in intervals
    ]
    overlaps = 0
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            start_i, end_i = parsed[i]
            start_j, end_j = parsed[j]
            if start_i < end_j and start_j < end_i:
                overlaps += 1
    return overlaps


def parallel_verdict(result: dict) -> bool:
    """至少两次派发且区间有重叠，才能证明单条意图内发生了真并行。"""
    return result.get("dispatch_count", 0) >= 2 and result.get("overlaps", 0) > 0


async def run_round(label: str, query: str) -> dict:
    session_id = f"parallel-{label}-{uuid.uuid4().hex[:6]}"
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=900, headers=AUTH_HEADERS) as client:
        stream,_ = await submit_stream(client,BASE_URL,session_id,BUYER_ID,query,timeout=900)
        stream.require_success()
        events=stream.frames
    wall_seconds = time.monotonic() - started

    intervals = _dispatch_intervals(events)
    return {
        "label": label,
        "wall_seconds": round(wall_seconds, 1),
        "dispatch_count": len(intervals),
        "overlaps": _count_overlaps(intervals),
        "intervals": intervals,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="真并行 fork 验证")
    parser.parse_args()

    print("== 并行版（prompt 引导同轮多派） ...", flush=True)
    parallel = await run_round("parallel", PARALLEL_QUERY)
    print(f"   wall={parallel['wall_seconds']}s dispatch={parallel['dispatch_count']} overlaps={parallel['overlaps']}")

    print("== 串行版（显式要求逐个来） ...", flush=True)
    serial = await run_round("serial", SERIAL_QUERY)
    print(f"   wall={serial['wall_seconds']}s dispatch={serial['dispatch_count']} overlaps={serial['overlaps']}")

    print("\n===== 结论 =====")
    for result in (parallel, serial):
        print(f"[{result['label']}] wall={result['wall_seconds']}s "
              f"派发 {result['dispatch_count']} 次，时间区间重叠 {result['overlaps']} 对")
        for agent, start, end in result["intervals"]:
            print(f"   - {agent}: {start} → {end}")
    if parallel_verdict(parallel):
        speedup = serial["wall_seconds"] / parallel["wall_seconds"] if parallel["wall_seconds"] else 0
        print(f"\n并行生效：并行版存在时间重叠，相对串行版加速 {speedup:.2f}x")
    else:
        print("\n并行未生效：并行版没有观察到派发时间重叠（模型可能仍逐个调用）")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
