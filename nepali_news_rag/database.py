from __future__ import annotations


class DatabaseClient:
    """Placeholder for SQL layer.

    This workspace is currently configured as RAG-only, so SQL capabilities are
    intentionally disabled until a structured data phase is introduced.
    """

    def execute(self, query: str):
        raise NotImplementedError(
            "SQL database path is disabled in this project scope (RAG-only)."
        )
