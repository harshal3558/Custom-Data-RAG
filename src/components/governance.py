"""
governance.py — Policy & Audit Governance Component

Maintains an append-only JSONL audit trail of every interaction
and enforces configurable policy rules over query/answer pairs.
"""
import os
import sys
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Optional
from src.exception import CustomException
from src.logger import logging


@dataclass
class GovernanceConfig:
    audit_log_dir: str = os.path.join("logs", "audit")
    audit_log_file: str = os.path.join("logs", "audit", "audit_trail.jsonl")
    policy_name: str = "default_rag_policy"
    require_sources: bool = True        # Answer must cite at least 1 source
    min_answer_length: int = 10         # Minimum non-trivial answer length (chars)


class GovernanceLayer:
    """
    Audit logging and policy enforcement for every RAG interaction.

    Usage
    -----
    gov = GovernanceLayer()
    gov.audit_log({"session_id": "abc", "query": "...", "answer": "...", "sources": [...]})
    passed, reason = gov.enforce_policy(query, answer, sources)
    report = gov.generate_report()
    """

    def __init__(self, config: GovernanceConfig = None):
        self.config = config or GovernanceConfig()
        os.makedirs(self.config.audit_log_dir, exist_ok=True)
        logging.info(f"GovernanceLayer initialised. Audit log: {self.config.audit_log_file}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit_log(self, event: dict) -> None:
        """
        Append an event record to the append-only JSONL audit trail.
        Automatically adds a UTC timestamp if not present.
        """
        try:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "policy": self.config.policy_name,
                **event,
            }
            with open(self.config.audit_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # Audit failures must never crash the main flow — just log them
            logging.error(f"GovernanceLayer: failed to write audit log: {e}")

    def enforce_policy(
        self,
        query: str,
        answer: str,
        sources: Optional[List[dict]] = None,
    ) -> Tuple[bool, str]:
        """
        Check whether a query/answer pair satisfies governance policy.

        Returns
        -------
        (True, "") if the policy passes.
        (False, reason_string) if a policy rule is violated.
        """
        try:
            sources = sources or []

            # Rule 1: Answer must not be empty / too short
            if not answer or len(answer.strip()) < self.config.min_answer_length:
                return False, "Answer is too short or empty — policy requires a substantive response."

            # Rule 2: If require_sources, answer must have at least one cited source
            if self.config.require_sources and len(sources) == 0:
                return False, "Policy requires at least one retrieved source to support the answer."

            return True, ""

        except Exception as e:
            raise CustomException(e, sys)

    def generate_report(self, last_n: int = 100) -> dict:
        """
        Read the last N audit log records and return a summary dict.

        Returns
        -------
        dict with keys: total_records, policy_pass_rate, unique_sessions,
                        sample_queries (up to 5).
        """
        try:
            if not os.path.exists(self.config.audit_log_file):
                return {"total_records": 0, "message": "No audit log found yet."}

            records: List[dict] = []
            with open(self.config.audit_log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            records = records[-last_n:]
            total = len(records)
            if total == 0:
                return {"total_records": 0, "message": "Audit log is empty."}

            passed = sum(1 for r in records if r.get("policy_passed") is True)
            sessions = {r.get("session_id") for r in records if r.get("session_id")}
            sample = [r.get("query", "") for r in records[:5]]

            return {
                "total_records": total,
                "policy_pass_rate": round(passed / total, 3),
                "unique_sessions": len(sessions),
                "sample_queries": sample,
            }

        except Exception as e:
            raise CustomException(e, sys)
