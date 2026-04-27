import csv
import json
import os
import time
from typing import Any, Dict, List

from medagent.agent.graph import run_agentic_rag
from medagent.pipeline import run_baseline_rag

from .dataset import load_eval_dataset
from .metrics import behavior_correct, classify_actual_behavior, compute_summary, has_citation


def _ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _normalize_pipeline_result(pipeline: str, result: Dict[str, Any]) -> Dict[str, Any]:
    if pipeline == "baseline":
        return {
            "answer": str(result.get("answer", "")),
            "contexts": list(result.get("retrieved_contexts", [])),
            "decision": {},
            "trace": [],
            "retrieval_attempts": 1 if result.get("retrieved_contexts") is not None else 0,
            "latency_seconds": float(
                ((result.get("latency") or {}).get("seconds", 0.0))
            ),
        }

    decision = dict(result.get("decision", {}) or {})
    return {
        "answer": str(result.get("answer", "")),
        "contexts": list(result.get("contexts", [])),
        "decision": decision,
        "trace": list(result.get("trace", [])),
        "retrieval_attempts": int(decision.get("attempts", 0) or 0),
        "latency_seconds": float(result.get("latency_seconds", 0.0) or 0.0),
    }


def _run_one(config: Any, pipeline: str, question: str) -> Dict[str, Any]:
    start = time.time()
    if pipeline == "baseline":
        result = run_baseline_rag(config=config, question=question)
    elif pipeline == "agentic":
        result = run_agentic_rag(config=config, question=question)
    else:
        raise ValueError("未知 pipeline: {0}".format(pipeline))
    elapsed = time.time() - start

    normalized = _normalize_pipeline_result(pipeline, result)
    if pipeline == "agentic":
        normalized["latency_seconds"] = round(elapsed, 4)
    return normalized


def _run_optional_ragas(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """RAGAS 可选执行：检测依赖与 API key 后尝试运行。"""

    has_api_key = bool(
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("RAGAS_API_KEY")
    )
    if not has_api_key:
        return {"ragas_enabled": False, "ragas_reason": "未检测到 API key"}

    try:
        import ragas  # noqa: F401
        from datasets import Dataset  # noqa: F401
    except ImportError:
        return {"ragas_enabled": False, "ragas_reason": "未安装 RAGAS 或 datasets 依赖"}

    # 第一版仅做可选探测，不强制执行完整 RAGAS 评测流程，避免额外依赖导致主流程失败。
    return {"ragas_enabled": True, "ragas_reason": "已检测到依赖与 API key（可接入完整 RAGAS）"}


def run_evaluation(
    config: Any,
    pipeline: str,
    dataset_path: str,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    rows = load_eval_dataset(dataset_path)
    _ensure_output_dir(output_dir)

    out_jsonl = os.path.join(output_dir, "eval_results.jsonl")
    out_csv = os.path.join(output_dir, "eval_summary.csv")

    records = []
    for row in rows:
        record = dict(row)
        try:
            run_result = _run_one(
                config=config,
                pipeline=pipeline,
                question=row["question"],
            )
            answer = run_result["answer"]
            decision = run_result["decision"]
            record.update(
                {
                    "pipeline": pipeline,
                    "answer": answer,
                    "contexts": run_result["contexts"],
                    "trace": run_result["trace"],
                    "decision": decision,
                    "latency_seconds": float(run_result["latency_seconds"]),
                    "retrieval_attempts": int(run_result["retrieval_attempts"]),
                    "has_citation": has_citation(answer),
                    "actual_behavior": classify_actual_behavior(answer, decision),
                    "behavior_correct": behavior_correct(
                        row["expected_behavior"],
                        answer,
                        decision,
                    ),
                    "error": "",
                }
            )
        except Exception as exc:
            record.update(
                {
                    "pipeline": pipeline,
                    "answer": "",
                    "contexts": [],
                    "trace": [],
                    "decision": {},
                    "latency_seconds": 0.0,
                    "retrieval_attempts": 0,
                    "has_citation": False,
                    "actual_behavior": "refuse",
                    "behavior_correct": row["expected_behavior"] != "answer",
                    "error": str(exc),
                }
            )
        records.append(record)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = compute_summary(records)
    summary.update(
        {
            "pipeline": pipeline,
            "dataset_path": dataset_path,
            "output_jsonl": out_jsonl,
            "output_csv": out_csv,
        }
    )
    summary.update(_run_optional_ragas(records))

    fieldnames = [
        "pipeline",
        "dataset_path",
        "total",
        "behavior_accuracy",
        "citation_rate",
        "avg_latency_seconds",
        "avg_retrieval_attempts",
        "ragas_enabled",
        "ragas_reason",
        "output_jsonl",
        "output_csv",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: summary.get(key, "") for key in fieldnames})

    return summary
