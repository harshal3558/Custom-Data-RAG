"""
gateway.py — LLM API Gateway Component

Single, authoritative place to create and configure the Groq LLM.
Provides retry logic with exponential back-off for transient API errors.
"""
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.exception import CustomException
from src.logger import logging

load_dotenv()


@dataclass
class GatewayConfig:
    model_name: str = ""           # Falls back to GROQ_MODEL_NAME env var
    temperature: float = 0.2
    max_retries: int = 3
    retry_base_delay: float = 1.0  # seconds; doubles on each retry
    request_timeout: int = 60      # seconds


class LLMGateway:
    """
    Centralised factory and invocation wrapper for the Groq LLM.

    Usage
    -----
    gateway = LLMGateway()
    llm     = gateway.get_llm()
    result  = gateway.invoke_with_retry(chain, {"query": "..."})
    """

    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self._llm: Optional[ChatGroq] = None
        self._load_api_key()
        logging.info(f"LLMGateway initialised (model={self._model_name()}).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_llm(self) -> ChatGroq:
        """Return a configured ChatGroq instance (singleton per gateway)."""
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    def invoke_with_retry(self, chain: Any, inputs: dict) -> Any:
        """
        Invoke a LangChain chain with exponential back-off retry.

        Parameters
        ----------
        chain  : Any runnable LangChain chain
        inputs : dict of chain input variables

        Returns
        -------
        Chain output on success.

        Raises
        ------
        CustomException after max_retries exhausted.
        """
        delay = self.config.retry_base_delay
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                return chain.invoke(inputs)
            except Exception as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    logging.warning(
                        f"LLM invoke attempt {attempt}/{self.config.max_retries} failed: {exc}. "
                        f"Retrying in {delay:.1f}s…"
                    )
                    time.sleep(delay)
                    delay *= 2  # exponential back-off
                else:
                    logging.error(f"All {self.config.max_retries} LLM invoke attempts failed.")

        raise CustomException(last_exc, sys)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_api_key(self) -> None:
        raw = os.getenv("GROQ_API_KEY", "")
        api_key = raw.strip().strip('"').strip("'")
        if not api_key:
            logging.error("GROQ_API_KEY not found in environment variables.")
        else:
            os.environ["GROQ_API_KEY"] = api_key
            logging.info(f"GROQ_API_KEY loaded (starts: {api_key[:4]}… ends: {api_key[-4:]})")
        self._api_key = api_key

    def _model_name(self) -> str:
        return self.config.model_name or os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

    def _build_llm(self) -> ChatGroq:
        try:
            return ChatGroq(
                api_key=self._api_key,
                groq_api_key=self._api_key,
                model_name=self._model_name(),
                temperature=self.config.temperature,
            )
        except Exception as e:
            raise CustomException(e, sys)
