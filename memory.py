"""
Memory Module
=============
Implements two-tier memory for DataSage:

  Short-term  — in-memory list of messages for the current session.
                Passed directly to the Claude API as conversation history.

  Long-term   — SQLite database that persists facts, user preferences,
                dataset metadata, and session summaries across restarts.
                Retrieved via simple keyword search and injected into the
                system prompt so the agent can reference past context.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional


class MemoryModule:
    """
    Two-tier memory: short-term (in-memory) and long-term (SQLite).
    """

    def __init__(self, db_path: str = "datasage_memory.db"):
        self.db_path = db_path
        # Short-term: current session conversation (list of {role, content} dicts)
        self.short_term: list[dict] = []
        # Hold a single connection so that :memory: databases work correctly
        # and file-based databases benefit from connection reuse.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------ #
    # Database initialisation                                              #
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                content   TEXT    NOT NULL,
                category  TEXT    DEFAULT 'general',
                timestamp TEXT    NOT NULL,
                session_id TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id            TEXT PRIMARY KEY,
                summary       TEXT,
                timestamp     TEXT NOT NULL,
                message_count INTEGER DEFAULT 0
            )
        """)
        self._conn.commit()

    def close(self) -> None:
        """Explicitly close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------ #
    # Short-term memory                                                    #
    # ------------------------------------------------------------------ #

    def add_message(self, role: str, content) -> None:
        """Append a message to short-term memory."""
        self.short_term.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """Return a copy of the current short-term conversation history."""
        return list(self.short_term)

    def clear_session(self) -> None:
        """Wipe short-term memory (start a fresh session)."""
        self.short_term = []

    def message_count(self) -> int:
        return len(self.short_term)

    # ------------------------------------------------------------------ #
    # Long-term memory                                                     #
    # ------------------------------------------------------------------ #

    def save(self, content: str, category: str = "general", session_id: Optional[str] = None) -> int:
        """
        Persist a piece of information to long-term memory.

        Parameters
        ----------
        content     : the text to remember
        category    : tag for organisation (e.g. 'dataset', 'preference', 'insight')
        session_id  : optional link to a session record

        Returns
        -------
        The row ID of the inserted record.
        """
        cursor = self._conn.execute(
            "INSERT INTO memories (content, category, timestamp, session_id) VALUES (?, ?, ?, ?)",
            (content, category, datetime.now().isoformat(), session_id),
        )
        self._conn.commit()
        return cursor.lastrowid

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Keyword search over long-term memories.
        Returns up to *limit* rows ordered by recency.
        """
        words = [w for w in query.lower().split() if len(w) > 2]
        if not words:
            return self._recent(limit)

        conditions = " OR ".join(["LOWER(content) LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words]

        rows = self._conn.execute(
            f"SELECT id, content, category, timestamp FROM memories "
            f"WHERE {conditions} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()

        return [{"id": r[0], "content": r[1], "category": r[2], "timestamp": r[3]} for r in rows]

    def _recent(self, limit: int = 10) -> list[dict]:
        """Return the *limit* most recent memories."""
        rows = self._conn.execute(
            "SELECT id, content, category, timestamp FROM memories ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"id": r[0], "content": r[1], "category": r[2], "timestamp": r[3]} for r in rows]

    def delete(self, memory_id: int) -> None:
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    def save_session_summary(self, session_id: str, summary: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (id, summary, timestamp, message_count) VALUES (?, ?, ?, ?)",
            (session_id, summary, datetime.now().isoformat(), self.message_count()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Context injection helpers                                            #
    # ------------------------------------------------------------------ #

    def build_memory_context(self, query: str = "") -> str:
        """
        Return a formatted block of relevant long-term memories suitable
        for injection into the system prompt.
        """
        memories = self.search(query, limit=6) if query else self._recent(6)
        if not memories:
            return ""

        lines = ["## Relevant context from previous sessions"]
        for m in memories:
            ts = m["timestamp"][:16].replace("T", " ")
            lines.append(f"- [{m['category']} | {ts}] {m['content']}")
        return "\n".join(lines)

    def all_stats(self) -> dict:
        """Return basic statistics about the memory store."""
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_cat = self._conn.execute(
            "SELECT category, COUNT(*) FROM memories GROUP BY category"
        ).fetchall()
        sessions = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return {
            "total_memories": total,
            "sessions": sessions,
            "by_category": {r[0]: r[1] for r in by_cat},
        }
