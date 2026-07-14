"""
security.py — Security Layer Component

Handles API-key validation, prompt-injection scanning, input sanitisation,
and session-ID hashing so raw identifiers are never persisted in logs.
"""
import os
import re
import sys
import hmac
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple
from src.exception import CustomException
from src.logger import logging


@dataclass
class SecurityConfig:
    api_key_env_var: str = "INTERNAL_API_KEY"    # Set in .env to enable API-key auth
    max_prompt_length: int = 4000
    injection_patterns: List[str] = field(default_factory=lambda: [
        r"(?i)ignore\s+(all\s+)?previous\s+instructions?",
        r"(?i)you\s+are\s+now\s+",
        r"(?i)forget\s+(everything|all)\s+",
        r"(?i)system\s*prompt\s*:",
        r"(?i)<\s*script[^>]*>",               # XSS via prompt
        r"(?i)\beval\s*\(",                     # code injection attempt
        r"(?i)os\.system|subprocess|__import__",# Python injection
    ])
    strip_html: bool = True


class SecurityLayer:
    """
    Request-level security checks for the RAG application.

    Usage
    -----
    sec = SecurityLayer()
    clean = sec.sanitise_input(raw_text)
    ok, reason = sec.scan_for_injection(clean)
    if not ok:
        abort(400, reason)
    """

    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self._compiled_patterns = [
            re.compile(p) for p in self.config.injection_patterns
        ]
        logging.info("SecurityLayer initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sanitise_input(self, text: str) -> str:
        """
        Strip control characters, HTML tags (optional), and
        collapse excessive whitespace from raw user input.
        """
        try:
            if not text:
                return text

            # Remove NULL bytes and control characters (keep \n, \t)
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

            # Strip HTML tags
            if self.config.strip_html:
                text = re.sub(r'<[^>]+>', '', text)

            # Collapse runs of whitespace into a single space, preserve newlines
            lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]
            text = '\n'.join(line for line in lines if line)

            return text.strip()

        except Exception as e:
            raise CustomException(e, sys)

    def scan_for_injection(self, text: str) -> Tuple[bool, str]:
        """
        Scan text for prompt-injection patterns.

        Returns
        -------
        (True, "")          — no injection detected
        (False, reason_str) — injection pattern found
        """
        try:
            for pattern in self._compiled_patterns:
                match = pattern.search(text)
                if match:
                    logging.warning(
                        f"SecurityLayer: injection pattern detected: '{pattern.pattern}' "
                        f"at position {match.start()}"
                    )
                    return False, "Potential prompt injection detected. Request blocked."

            if len(text) > self.config.max_prompt_length:
                return False, (
                    f"Input exceeds maximum allowed length of "
                    f"{self.config.max_prompt_length} characters."
                )

            return True, ""

        except Exception as e:
            raise CustomException(e, sys)

    def validate_api_key(self, provided_key: str) -> bool:
        """
        Constant-time comparison of provided key against the env-var secret.
        Returns True if keys match OR if no internal key is configured
        (open-access mode).
        """
        try:
            expected = os.getenv(self.config.api_key_env_var, "").strip()
            if not expected:
                # No internal key configured — open access (development mode)
                return True

            return hmac.compare_digest(
                provided_key.encode("utf-8"),
                expected.encode("utf-8"),
            )
        except Exception as e:
            raise CustomException(e, sys)

    def hash_session_id(self, raw_id: str) -> str:
        """
        Return a SHA-256 hex digest of the session ID.
        Use this hash in logs/audit trails instead of the raw session ID.
        """
        return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

    def get_blocked_response(self, reason: str) -> dict:
        """Standard JSON-safe response when a security check fails."""
        return {
            "answer": f"🔒 Request blocked by security layer: {reason}",
            "sources": [],
            "security_blocked": True,
        }
