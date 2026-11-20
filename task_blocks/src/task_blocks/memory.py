from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Any

from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from task_blocks.config import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIMS = 768

MEMORY_NAMESPACE = "memories"

_lock = threading.Lock()
_checkpointer: SqliteSaver | None = None
_store: SqliteStore | None = None


def memory_namespace(user_id: str) -> tuple[str, str]:
    return (MEMORY_NAMESPACE, user_id or "anonymous")


def _connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(path, check_same_thread=False, isolation_level=None)


def sqlite_supports_extensions() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        return hasattr(conn, "enable_load_extension")
    finally:
        conn.close()


def get_embeddings() -> OllamaEmbeddings:
    settings = get_settings()
    return OllamaEmbeddings(
        model=settings.embedding_model, base_url=settings.ollama_base_url
    )


def get_checkpointer() -> SqliteSaver:
    global _checkpointer
    with _lock:
        if _checkpointer is None:
            settings = get_settings()
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            path = str(settings.checkpointer_path)
            checkpointer = SqliteSaver(_connect(path))
            checkpointer.setup()
            _checkpointer = checkpointer
            logger.info("checkpointer_ready", extra={"path": path})
        return _checkpointer


def get_store() -> SqliteStore:
    global _store
    with _lock:
        if _store is None:
            settings = get_settings()
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            path = str(settings.store_path)
            vector_search = sqlite_supports_extensions()

            index = (
                {
                    "dims": EMBEDDING_DIMS,
                    "embed": get_embeddings(),
                    "fields": ["content"],
                }
                if vector_search
                else None
            )
            if not vector_search:
                logger.error(
                    "store_without_vector_index",
                    extra={
                        "reason": (
                            "this Python build's sqlite3 cannot load extensions, "
                            "so sqlite-vec is unavailable; long-term memory "
                            "search will not be semantic"
                        ),
                    },
                )

            store = SqliteStore(_connect(path), index=index)
            store.setup()
            _store = store
            logger.info(
                "store_ready",
                extra={
                    "path": path,
                    "vector_search": vector_search,
                    "embedding_model": settings.embedding_model if vector_search else None,
                    "dims": EMBEDDING_DIMS if vector_search else None,
                },
            )
        return _store


def reset_memory() -> None:
    global _checkpointer, _store
    with _lock:
        for component in (_checkpointer, _store):
            conn = getattr(component, "conn", None)
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:  # pragma: no cover - best effort
                    pass
        _checkpointer = None
        _store = None


_KEYWORD_SCAN_LIMIT = 100

_STOPWORDS = frozenset(
    """a an and are as at be but by do does did for from had has have how i if in
    is it its me my of on or our so that the their them then there these they this
    to was we were what when where which who why will with you your""".split()
)


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in "".join(
            char.lower() if char.isalnum() else " " for char in text
        ).split()
        if len(word) > 2 and word not in _STOPWORDS
    }


def _extract_content(value: Any) -> str:
    if isinstance(value, dict):
        content = value.get("content") or value.get("memory") or ""
    else:
        content = value
    return content.strip() if isinstance(content, str) else str(content).strip()


def search_memories(user_id: str, query: str, *, limit: int = 5) -> list[str]:
    if not query.strip():
        return []

    semantic = sqlite_supports_extensions()
    scan_limit = limit if semantic else _KEYWORD_SCAN_LIMIT

    try:
        items = get_store().search(
            memory_namespace(user_id), query=query, limit=scan_limit
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "memory_search_failed",
            extra={"error": f"{type(exc).__name__}: {exc}", "user_id": user_id},
        )
        return []

    memories = [text for text in (_extract_content(i.value) for i in items) if text]

    if not semantic and memories:
        query_tokens = _tokens(query)
        scored = [
            (len(query_tokens & _tokens(text)), index, text)
            for index, text in enumerate(memories)
        ]
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        memories = [text for _, _, text in scored][:limit]

    logger.info(
        "memory_search",
        extra={
            "user_id": user_id,
            "retrieved": len(memories),
            "limit": limit,
            "semantic": semantic,
        },
    )
    return memories


def health() -> dict[str, Any]:
    settings = get_settings()
    status: dict[str, Any] = {
        "data_dir": str(settings.data_dir),
        "checkpointer": "unavailable",
        "store": "unavailable",
        "vector_search": sqlite_supports_extensions(),
    }
    try:
        get_checkpointer()
        status["checkpointer"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["checkpointer_error"] = f"{type(exc).__name__}: {exc}"
    try:
        get_store()
        status["store"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["store_error"] = f"{type(exc).__name__}: {exc}"
    return status
