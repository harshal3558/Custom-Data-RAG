"""
guardrails.py — Input / Output Guardrails Component

Validates and sanitises queries before they reach the LLM, and
filters responses before they are returned to the user.

Checks performed:
  Input  : max length, PII redaction (email, phone), profanity/injection blocklist
  Output : max length truncation, PII redaction
"""
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Tuple
from src.exception import CustomException
from src.logger import logging


# ------------------------------------------------------------------
# Custom Exception
# ------------------------------------------------------------------

class GuardrailViolation(ValueError):
    """Raised when a guardrail rule is violated."""


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class GuardrailsConfig:
    max_input_chars: int = 2000
    max_output_chars: int = 8000
    enable_pii_redaction: bool = True
    banned_keywords: List[str] = field(default_factory=lambda: [
        # Prompt-injection attempts
        "ignore previous instructions",
        "ignore all instructions",
        "disregard your instructions",
        "you are now",
        "act as",
        "jailbreak",
        "dan mode",
        # Explicit harmful requests
        "how to make a bomb",
        "how to hack",
        "generate malware",
        "write a virus",
    ])


# ------------------------------------------------------------------
# PII Patterns
# ------------------------------------------------------------------

_PII_PATTERNS = [
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b'), "[EMAIL REDACTED]"),
    (re.compile(r'\b(?:\+?91[-\s]?)?[6-9]\d{9}\b'), "[PHONE REDACTED]"),        # Indian mobile
    (re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'), "[PHONE REDACTED]"),     # US/intl phone
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "[SSN REDACTED]"),                   # US SSN
    (re.compile(r'\b(?:\d[ -]?){13,16}\b'), "[CARD REDACTED]"),                 # Credit card
]


# ------------------------------------------------------------------
# Main Class
# ------------------------------------------------------------------

class Guardrails:
    """
    Usage
    -----
    guard = Guardrails()
    safe_query = guard.check_input(user_query)      # raises GuardrailViolation on breach
    safe_answer = guard.check_output(llm_answer)
    """

    def __init__(self, config: GuardrailsConfig = None):
        self.config = config or GuardrailsConfig()
        logging.info("Guardrails initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_input(self, query: str) -> str:
        """
        Validate and sanitise a user query.

        Returns sanitised query string.
        Raises GuardrailViolation if a hard rule is breached.
        """
        try:
            if not query or not query.strip():
                raise GuardrailViolation("Query must not be empty.")

            # Length check
            if len(query) > self.config.max_input_chars:
                raise GuardrailViolation(
                    f"Query exceeds maximum allowed length of {self.config.max_input_chars} characters."
                )

            # Banned keyword check (case-insensitive)
            lower = query.lower()
            for keyword in self.config.banned_keywords:
                if keyword in lower:
                    logging.warning(f"Guardrail blocked input containing banned keyword: '{keyword}'")
                    raise GuardrailViolation(
                        f"Your message was flagged by our content policy and cannot be processed."
                    )

            # PII redaction
            sanitised = self._redact_pii(query) if self.config.enable_pii_redaction else query

            return sanitised

        except GuardrailViolation:
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def check_output(self, answer: str) -> str:
        """
        Sanitise an LLM response before returning it to the user.

        Returns sanitised answer string (never raises GuardrailViolation).
        """
        try:
            if not answer:
                return answer

            # Truncate overly long responses
            if len(answer) > self.config.max_output_chars:
                answer = answer[: self.config.max_output_chars] + "… [response truncated]"
                logging.warning("LLM output truncated by guardrail (exceeded max_output_chars).")

            # PII redaction on output as well
            if self.config.enable_pii_redaction:
                answer = self._redact_pii(answer)

            return answer

        except Exception as e:
            raise CustomException(e, sys)

    def get_violation_response(self, reason: str) -> dict:
        """Standard JSON-safe error response for guardrail violations."""
        return {
            "answer": f"⚠️ Your request could not be processed: {reason}",
            "sources": [],
            "guardrail_blocked": True,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _redact_pii(self, text: str) -> str:
        for pattern, replacement in _PII_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
