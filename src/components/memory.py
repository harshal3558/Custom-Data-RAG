"""
memory.py — Conversation Memory Component

Manages per-session chat history with a configurable turn limit.
Sessions are stored in-process (in a dict). For multi-process deployments
swap the backing store to Redis or a database.
"""
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional
from src.exception import CustomException
from src.logger import logging


@dataclass
class MemoryConfig:
    max_turns: int = 10          # Maximum (user + assistant) turn pairs to keep
    system_prefix: str = (
        "You are GroqRAG Turbo, an advanced AI specialized in analysing PDF documents."
    )


class ConversationMemory:
    """
    Lightweight in-process session memory.

    Usage
    -----
    memory = ConversationMemory()
    memory.add_turn("session-123", "user", "What is this doc about?")
    memory.add_turn("session-123", "assistant", "It is about ...")
    history = memory.get_history("session-123")   # list[dict]
    """

    def __init__(self):
        self.config = MemoryConfig()
        self._store: Dict[str, List[Dict[str, str]]] = {}
        logging.info("ConversationMemory initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a single message to the session history."""
        try:
            if session_id not in self._store:
                self._store[session_id] = []
            self._store[session_id].append({"role": role, "content": content})
            self._trim(session_id)
        except Exception as e:
            raise CustomException(e, sys)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return all stored messages for a session (oldest first)."""
        return list(self._store.get(session_id, []))

    def get_history_text(self, session_id: str) -> str:
        """Return history as a formatted string suitable for prompt injection."""
        msgs = self.get_history(session_id)
        return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in msgs)

    def clear(self, session_id: str) -> None:
        """Remove all history for a session."""
        self._store.pop(session_id, None)
        logging.info(f"Memory cleared for session: {session_id}")

    def session_count(self) -> int:
        """Return number of active sessions in memory."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim(self, session_id: str) -> None:
        """Keep only the most recent max_turns pairs."""
        history = self._store[session_id]
        max_messages = self.config.max_turns * 2  # each pair = user + assistant
        if len(history) > max_messages:
            self._store[session_id] = history[-max_messages:]
