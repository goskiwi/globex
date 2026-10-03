"""演示启停复用已有控制器，按顺序启动、等待真实健康，失败后收回服务。"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def session(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts/bge_session.sh", scripts)
    for name, label in [("bge_remote.sh", "remote"), ("bge_tunnel.sh", "tunnel")]:
        (scripts / name).write_text(
            f'#!/bin/sh\nprintf "{label}:%s\\n" "$1" >> "$EVENTS"\n'
            + ('if [ "$1" = start ] && [ "${TUNNEL_FAIL:-0}" = 1 ]; then exit 1; fi\n' if label == "tunnel" else "")
        )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    curl = binaries / "curl"
    curl.write_text('#!/bin/sh\nprintf "health\\n" >> "$EVENTS"\nprintf \'{"status":"ok"}\\n\'\n')
    curl.chmod(0o755)
    events = tmp_path / "events.txt"
    env = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}", "EVENTS": str(events)}

    def run(action, **extra):
        result = subprocess.run(["sh", str(scripts / "bge_session.sh"), action], env={**env, **extra},
                                capture_output=True, text=True, timeout=10)
        return result, events.read_text().splitlines()

    return run


def test_start_waits_for_health_and_keeps_models_for_this_session(session):
    result, events = session("start")
    assert result.returncode == 0
    assert events == ["remote:start", "tunnel:start", "health"]


def test_stop_closes_remote_models_then_local_tunnel(session):
    result, events = session("stop")
    assert result.returncode == 0
    assert events == ["remote:stop", "tunnel:stop"]


def test_failed_tunnel_start_releases_project_models(session):
    result, events = session("start", TUNNEL_FAIL="1")
    assert result.returncode != 0
    assert events == ["remote:start", "tunnel:start", "remote:stop", "tunnel:stop"]


def test_docker_commands_include_model_lifecycle_in_order():
    for action, model_action, compose_action, model_first in [
        ("docker-up", "start", "up -d --build", True),
        ("docker-stop", "stop", " stop", False),
    ]:
        result = subprocess.run(["make", "-n", action], cwd=ROOT, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        model_pos = result.stdout.index(f"sh scripts/bge_session.sh {model_action}")
        compose_pos = result.stdout.index(compose_action)
        assert (model_pos < compose_pos) == model_first
