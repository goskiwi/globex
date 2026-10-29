"""GEPA 离线改进：候选通过真实应用、主子图和交付执行，只导出待审核版本。

运行：uv run --extra optimization python -m scripts.optimize_prompts --output .pytest_cache/gepa-run
候选与经验均在隔离库试验，正式发布沿用 prompt_release / capabilities 的已有审核入口。
"""

from __future__ import annotations
import argparse
from copy import deepcopy
import asyncio
import hashlib
import json
import math
import re
from uuid import uuid4
from pathlib import Path
import yaml


def validate_cases(cases):
    from scripts.eval_regression import build_judge_rubric
    from scripts.eval.http_actions import validate_http_actions

    seen_ids, seen_queries, families = set(), set(), {}
    for case in cases:
        if (
            not isinstance(case.get("id"), str)
            or not case["id"]
            or re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}',case['id']) is None
            or case.get("split") not in {"train", "val", "holdout"}
            or not isinstance(case.get("rubric"), dict)
            or not isinstance(case.get("family"), str) or not case["family"].strip()
        ):
            raise ValueError("样本需要 id / family / train,val,holdout split / rubric")
        queries = case.get("queries")
        if not isinstance(queries, list) or not queries or any(not isinstance(q,str) or not q.strip() for q in queries):
            raise ValueError("样本需要完整、非空的 queries")
        normalized = re.sub(r"\s+", "", "\n".join(queries)).casefold()
        family = case["family"]
        if family in families and families[family] != case["split"]:
            raise ValueError("同一需求家族及其改写不能跨训练、选择和留出集")
        families[family] = case["split"]
        if any(build_judge_rubric(case["rubric"], case.get("deterministic")).values()):
            raise ValueError("优化任务必须用业务断言评分；主观评价另行审阅")
        validate_http_actions(case)
        if case["id"] in seen_ids or normalized in seen_queries:
            raise ValueError("样本 ID 或需求重复；禁止训练与留出泄漏")
        seen_ids.add(case["id"])
        seen_queries.add(normalized)
    if {c["split"] for c in cases} != {"train", "val", "holdout"}:
        raise ValueError("必须分别提供训练、选择与留出集")


def reserve_output(output):
    # 新建专属目录，不覆盖既有报告、生产数据或上次候选。
    output.mkdir(parents=True, exist_ok=False)


def optimize_candidate(
    seed, cases, evaluator, output, *, max_calls=24, proposer=None, reflection_lm=None
):
    from gepa import optimize
    from gepa.core.adapter import EvaluationBatch

    validate_cases(cases)
    if not isinstance(seed, str) or not seed.strip() or not 4 <= max_calls <= 1000:
        raise ValueError("候选不能为空，评估预算为 4 到 1000 次")
    reserve_output(output)
    records = []
    candidate_instructions = {}

    class Adapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            text = candidate.get("instruction")
            if (
                set(candidate) != {"instruction"}
                or not isinstance(text, str)
                or not 1 <= len(text) <= 32000
            ):
                raise ValueError("候选只能修改 instruction 正文")
            results = []
            candidate_id = hashlib.sha256(text.encode()).hexdigest()
            candidate_instructions[candidate_id] = text
            for case in batch:
                result = evaluator(case, text)
                score = result.get("optimization_score", result.get("score"))
                if (
                    type(score) not in (int, float)
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError("评测分数必须为 0 到 1 的有限值")
                if result.get("violations"):
                    score = 0.0
                result = {**result, "score": score}
                results.append(result)
                records.append(
                    {
                        "id": case["id"],
                        "split": case["split"],
                        "candidate_hash": candidate_id,
                        **result,
                    }
                )
            traces = [
                {
                    "Inputs": c["queries"],
                    "Outputs": r.get("output"),
                    "Feedback": r.get("feedback"),
                }
                for c, r in zip(batch, results)
            ]
            return EvaluationBatch(
                outputs=results,
                scores=[r["score"] for r in results],
                trajectories=traces if capture_traces else None,
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return {key: eval_batch.trajectories for key in components_to_update}

    adapter = Adapter()
    reflection_errors = []

    def checked_reflection(messages):
        try:
            return reflection_lm(messages)
        except Exception as error:
            reflection_errors.append(type(error).__name__)
            raise

    try:
        result = optimize(
            seed_candidate={"instruction": seed},
            trainset=[c for c in cases if c["split"] == "train"],
            valset=[c for c in cases if c["split"] == "val"],
            adapter=adapter,
            max_metric_calls=max_calls,
            reflection_minibatch_size=min(2, sum(c["split"] == "train" for c in cases)),
            custom_candidate_proposer=proposer,
            reflection_lm=checked_reflection if reflection_lm else None,
            run_dir=str(output / "engine"),
            seed=20260918,
            use_wandb=False,
            use_mlflow=False,
            cache_evaluation=False,
            skip_perfect_score=False,
            raise_on_exception=True,
        )
        if reflection_errors:
            raise RuntimeError(
                "反思调用失败，不能把未产生有效候选算作优化完成："
                + ",".join(reflection_errors)
            )
        candidate = result.best_candidate["instruction"]
        # 留出仅在搜索结束后执行，按场景交错评估，不反馈给优化器。
        baseline_scores = []
        candidate_scores = []
        task_outcomes = []
        for i, case in enumerate(c for c in cases if c["split"] == "holdout"):
            variants = [("baseline", seed), ("candidate", candidate)]
            if i % 2:
                variants.reverse()
            pair = {
                name: adapter.evaluate([case], {"instruction": text})
                for name, text in variants
            }
            baseline_scores.append(pair["baseline"].scores[0])
            candidate_scores.append(pair["candidate"].scores[0])
            task_outcomes.append({'id':case['id'], **{name:batch.outputs[0].get('task_score', batch.scores[0])
                                                     for name,batch in pair.items()}})
        report = {
            "scope": "production_task_runtime",
            "metric": "passed_task_per_actual_execution_step",
            "metric_definition": "任务通过且候选实际执行时为1/(1+真实模型请求数+工具调用数)，否则为0；非货币成本",
            "publication": "manual_review_required",
            "candidate": candidate,
            "candidate_changed": candidate != seed,
            "comparison_kind": "candidate_vs_baseline"
            if candidate != seed
            else "baseline_repeatability",
            "interpretation": "仍需完整 Agent 发布门禁"
            if candidate != seed
            else "正文未变化，分数差异仅代表重复生成波动，不是优化收益",
            "max_optimization_metric_calls": max_calls,
            "total_evaluations_including_holdout": len(records),
            "dataset_sha256": hashlib.sha256(
                json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            "holdout": {
                "baseline_scores": baseline_scores,
                "candidate_scores": candidate_scores,
                "task_outcomes": task_outcomes,
                "paired_mean_delta": sum(
                    b - a for a, b in zip(baseline_scores, candidate_scores)
                )
                / len(baseline_scores),
            },
            "holdout_cases": [case['id'] for case in cases if case['split'] == 'holdout'],
            "full_agent_release_gate": "NOT_RUN",
            "records": records,
            "candidate_instructions": candidate_instructions,
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        return report
    except BaseException as error:
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "status": "ERROR",
                    "error_type": type(error).__name__,
                    "failed_execution": getattr(error, 'execution_ref', None),
                    "records": records,
                    "candidate_instructions": candidate_instructions,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise


def skill_candidate(source, body, report_path):
    from app.infrastructure.capability_registry import validate

    original = validate(source)
    if original["kind"] != "skill":
        raise ValueError("此入口只优化公共 Skill 正文")
    result = {
        **original,
        "body": body,
        "version": "gepa-" + hashlib.sha256(body.encode()).hexdigest()[:16],
        "evidence": [
            *original["evidence"][:11],
            {
                "reference": str(report_path),
                "summary": "隔离真实任务执行与留出对照；候选尚未正式审核发布。",
            },
        ],
    }
    return validate(result)


def export_candidate(output, document, role, report, skill_source=None):
    from scripts.eval.task_evaluator import write_json
    from app.infrastructure.prompt_registry import read_prompt
    if skill_source is not None:
        write_json(output / 'candidate-skill.json', skill_candidate(skill_source,report['candidate'],output/'report.json'))
    else:
        candidate = deepcopy(document)
        target = candidate['main_agent'] if role == 'main' else candidate['sub_agents'][role]
        target['system_prompt'] = report['candidate']
        path = output / 'candidate.yml'
        path.write_text(yaml.safe_dump(candidate,allow_unicode=True,sort_keys=False))
        read_prompt(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path("eval/optimization_search.jsonl")
    )
    parser.add_argument(
        "--prompt", type=Path, default=Path("app/application/prompts/globex.yml")
    )
    parser.add_argument(
        "--public-skill",
        type=Path,
        help="公共 Skill 规范 JSON，仅优化 body，不连接个人 Skill 数据库",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-calls", type=int, default=24)
    parser.add_argument("--role", choices=["main", "search", "trade"], default="search")
    parser.add_argument("--isolated-knowledge", action="store_true")
    parser.add_argument("--experience", action="store_true", help="从本次对照提炼经验，并在留出任务验证实际加载与业务结果")
    args = parser.parse_args()
    from app.infrastructure.settings import load_settings
    from app.infrastructure.prompt_registry import read_prompt

    document, source_hash = read_prompt(args.prompt)
    target = document["main_agent"] if args.role == "main" else document["sub_agents"][args.role]
    seed = target["system_prompt"]
    skill_source = None
    if args.public_skill:
        from app.infrastructure.capability_registry import validate

        skill_source = validate(json.loads(args.public_skill.read_text()))
        if skill_source["kind"] != "skill":
            parser.error("--public-skill 仅支持 kind=skill")
        seed = skill_source["body"]
    cases = [
        json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()
    ]
    with asyncio.Runner() as runner:
        from scripts.eval.task_evaluator import TaskEvaluator
        evaluator = TaskEvaluator(load_settings(), runner, document=document, output=args.output,
            role=args.role, skill_source=skill_source, isolated_knowledge=args.isolated_knowledge)
        report = None
        try:
            report = optimize_candidate(
                seed,
                cases,
                evaluator,
                args.output,
                max_calls=args.max_calls,
                reflection_lm=evaluator.reflect,
            )
            export_candidate(args.output,document,args.role,report,skill_source)
            from scripts.eval.task_evaluator import write_json
            write_json(args.output / 'source.json', {'prompt_sha256':source_hash,
                'skill_sha256':hashlib.sha256(args.public_skill.read_bytes()).hexdigest() if args.public_skill else None,
                'scope':args.role,'model':evaluator.settings.llm_model})
            if args.experience:
                from scripts.improvement.experience import distill_experience, verify_reuse
                documents = distill_experience(report['records'], evaluator.reflect, output=args.output,
                    version='experiment-' + uuid4().hex[:12], instructions=report['candidate_instructions'])
                reuse = verify_reuse(cases,evaluator,seed,documents,args.output)
                report['experience'] = {'drafts':len(documents), 'reuse':reuse,
                    'publication':'isolated_only; existing_capability_review_required'}
        except BaseException as error:
            if report is not None:
                report['experience_failure'] = {'type':type(error).__name__,
                    'failed_execution':getattr(error,'execution_ref',None)}
            raise
        finally:
            from scripts.eval.task_evaluator import write_json
            usage = evaluator.usage_summary()
            if evaluator.records or report is not None:
                write_json(args.output / 'execution-summary.json', usage)
            if report is not None:
                report['reflection_usage'] = evaluator.reflection_usage
                report['experiment_usage'] = usage
                write_json(args.output / 'report.json', report)
            runner.run(evaluator.close())
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"candidate", "records", "candidate_instructions", "reflection_usage", "experience"}},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
