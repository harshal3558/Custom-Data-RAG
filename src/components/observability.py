"""
observability.py — Observability & Metrics Component

Records per-request metrics to a structured JSONL file, exposes
health-check logic for the /health endpoint, and computes
summary statistics (avg latency, p95, error rate, and LLM-as-a-judge scores)
over recent records.
"""
import os
import sys
import json
import mlflow
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict
from src.exception import CustomException
from src.logger import logging


@dataclass
class ObservabilityConfig:
    metrics_dir: str = os.path.join("logs", "observability")
    metrics_file: str = os.path.join("logs", "observability", "metrics.jsonl")
    slow_query_threshold_sec: float = 5.0   # Queries above this latency are flagged
    mlflow_experiment: str = "RAG_Observability"


class ObservabilityLayer:
    """
    Structured metrics collection, health checks, and summary reporting.

    Usage
    -----
    obs = ObservabilityLayer()
    obs.record_request(session_id="abc", query="...", answer="...",
                       latency=1.2, num_docs=4, avg_score=0.76, eval_scores={...})
    health = obs.health_check()
    summary = obs.get_summary(n=100)
    """

    def __init__(self, config: ObservabilityConfig = None):
        self.config = config or ObservabilityConfig()
        os.makedirs(self.config.metrics_dir, exist_ok=True)
        logging.info(f"ObservabilityLayer initialised. Metrics: {self.config.metrics_file}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_request(
        self,
        session_id: str,
        query: str,
        answer: str,
        latency: float,
        num_docs: int = 0,
        avg_score: float = 0.0,
        error: Optional[str] = None,
        eval_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """Append a single request record to the metrics JSONL file."""
        try:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "query_length": len(query),
                "answer_length": len(answer) if answer else 0,
                "latency_sec": round(latency, 4),
                "num_docs_retrieved": num_docs,
                "avg_retrieval_score": round(avg_score, 4),
                "is_slow": latency > self.config.slow_query_threshold_sec,
                "is_error": error is not None,
                "error": error,
                "eval_scores": eval_scores or {},
            }

            # Log to MLflow if active run is open
            try:
                if eval_scores and mlflow.active_run():
                    for score_name, score_val in eval_scores.items():
                        if score_val is not None:
                            mlflow.log_metric(f"live_{score_name}", score_val)
            except Exception as me:
                logging.warning(f"Could not log live LLM-as-a-judge score to MLflow: {me}")

            with open(self.config.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            if record["is_slow"]:
                logging.warning(
                    f"Slow query detected for session {session_id}: "
                    f"{latency:.2f}s > threshold {self.config.slow_query_threshold_sec}s"
                )

        except Exception as e:
            # Observability failures must not crash the main flow
            logging.error(f"ObservabilityLayer: failed to record metrics: {e}")

    def health_check(self) -> dict:
        """
        Run health checks and return a status dict.

        Checks:
          - vector_store  : data/vector_store directory exists
          - audit_log     : logs/audit directory exists
          - metrics_log   : metrics JSONL file exists
          - mlflow        : can set an MLflow experiment without error
        """
        status = {}

        # Vector store
        vs_path = os.path.join("data", "vector_store")
        status["vector_store"] = "ok" if os.path.exists(vs_path) else "missing"

        # Audit log dir
        audit_path = os.path.join("logs", "audit")
        status["audit_log"] = "ok" if os.path.exists(audit_path) else "missing"

        # Metrics log
        status["metrics_log"] = (
            "ok" if os.path.exists(self.config.metrics_file) else "no_records_yet"
        )

        # MLflow connectivity
        try:
            mlflow.set_experiment(self.config.mlflow_experiment)
            status["mlflow"] = "ok"
        except Exception as e:
            status["mlflow"] = f"error: {e}"

        status["overall"] = (
            "healthy"
            if all(v in ("ok", "no_records_yet") for v in status.values())
            else "degraded"
        )

        logging.info(f"Health check result: {status}")
        return status

    def get_summary(self, n: int = 100) -> dict:
        """
        Read the last *n* metrics records and return aggregate statistics.

        Returns
        -------
        dict with metrics and average LLM-as-a-judge score breakdown.
        """
        try:
            if not os.path.exists(self.config.metrics_file):
                return {"total_requests": 0, "message": "No metrics recorded yet."}

            records = []
            with open(self.config.metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            records = records[-n:]
            total = len(records)
            if total == 0:
                return {"total_requests": 0, "message": "Metrics file is empty."}

            latencies = sorted(r["latency_sec"] for r in records)
            errors = sum(1 for r in records if r.get("is_error"))
            slow = sum(1 for r in records if r.get("is_slow"))
            avg_docs = sum(r.get("num_docs_retrieved", 0) for r in records) / total

            # Compute LLM as a Judge score aggregates
            judge_metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "answer_correctness",
                "safety_score"
            ]
            judge_sums = {m: 0.0 for m in judge_metrics}
            judge_counts = {m: 0 for m in judge_metrics}

            for r in records:
                evals = r.get("eval_scores", {})
                for m in judge_metrics:
                    val = evals.get(m)
                    if val is not None:
                        judge_sums[m] += float(val)
                        judge_counts[m] += 1

            avg_judge_scores = {}
            for m in judge_metrics:
                if judge_counts[m] > 0:
                    avg_judge_scores[m] = round(judge_sums[m] / judge_counts[m], 4)
                else:
                    avg_judge_scores[m] = None

            p95_idx = int(0.95 * total) - 1
            p95 = latencies[max(p95_idx, 0)]

            return {
                "total_requests": total,
                "error_rate": round(errors / total, 3),
                "avg_latency_sec": round(sum(latencies) / total, 3),
                "p95_latency_sec": round(p95, 3),
                "slow_query_count": slow,
                "avg_docs_retrieved": round(avg_docs, 2),
                "avg_llm_judge_scores": avg_judge_scores
            }

        except Exception as e:
            raise CustomException(e, sys)
