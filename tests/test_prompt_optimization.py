"""GEPA 离线工作流：隔离留出集、真实引擎、只导出候选。"""

import json
from pathlib import Path
import pytest


def test_split_leakage_and_output_overwrite_are_rejected(tmp_path):
    from scripts.optimize_prompts import validate_cases, reserve_output

    cases = [
        {"id": str(i), "split": s, "family":str(i), "queries": [q], "rubric": {"p0":[],"p1":[],"p2":[]}, "deterministic":{}}
        for i, (s, q) in enumerate(
            [("train", "预算300背包"), ("val", "预算300背包"), ("holdout", "杯子")]
        )
    ]
    with pytest.raises(ValueError, match="重复"):
        validate_cases(cases)
    reserve_output(tmp_path / "out")
    with pytest.raises(FileExistsError):
        reserve_output(tmp_path / "out")


def test_gepa_runs_real_engine_and_keeps_holdout_out_of_reflection(tmp_path):
    pytest.importorskip("gepa")
    from scripts.optimize_prompts import optimize_candidate

    calls = []
    cases = [
        {"id": s, "split": s, "family":s, "queries": [s], "rubric": {"p0":[],"p1":[],"p2":[]}, "deterministic":{}}
        for s in ["train", "val", "holdout"]
    ]

    def evaluate(case, text):
        calls.append((case["id"], text))
        return {
            "score": int("改进" in text),
            "feedback": "需要明确保留预算",
            "violations": [],
            "usage": [],
        }

    reflected = []

    def propose(candidate, dataset, components):
        reflected.extend(str(dataset))
        assert "holdout" not in str(dataset)
        return {"instruction": "改进版"}

    report = optimize_candidate(
        "基线", cases, evaluate, tmp_path / "run", max_calls=8, proposer=propose
    )
    assert report["candidate"] == "改进版"
    assert report["holdout"]["baseline_scores"] == [0]
    assert report["holdout"]["candidate_scores"] == [1]
    assert report["publication"] == "manual_review_required"
    assert report["scope"] == "production_task_runtime"
    assert (tmp_path / "run/report.json").is_file()


def test_task_dataset_has_no_cross_split_family_leakage():
    from scripts.optimize_prompts import validate_cases
    rows=[json.loads(line) for line in Path('eval/optimization_search.jsonl').read_text().splitlines()]
    validate_cases(rows)
    rows[1]['family']=rows[-1]['family']
    with pytest.raises(ValueError,match='家族'):
        validate_cases(rows)

def test_public_skill_export_preserves_permissions_and_enters_draft_only(tmp_path):
    from scripts.optimize_prompts import skill_candidate
    from tests.test_capability_registry import document
    from app.infrastructure.capability_registry import CapabilityRegistry

    source = document()
    candidate = skill_candidate(
        source, "先核对预算，不擅自扩大国家范围。", tmp_path / "report.json"
    )
    assert candidate["allowed_tools"] == source["allowed_tools"]
    assert (
        candidate["scope"] == source["scope"]
        and candidate["version"] != source["version"]
    )
    registry = CapabilityRegistry(tmp_path / "registry.db")
    saved = registry.import_draft(candidate, author="isolated-test")
    assert saved["state"] == "draft"
    assert source["body"] != candidate["body"]
