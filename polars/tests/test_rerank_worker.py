import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# Provide lightweight stubs for external dependencies so we can import the module
fake_httpx = types.ModuleType("httpx")


class _DummyHTTPError(Exception):
    pass


class _DummyTimeout(Exception):
    pass


fake_httpx.Client = object
fake_httpx.TimeoutException = _DummyTimeout
fake_httpx.HTTPError = _DummyHTTPError
sys.modules.setdefault("httpx", fake_httpx)

fake_psycopg = types.ModuleType("psycopg")
fake_psycopg.Connection = object
fake_psycopg.OperationalError = Exception
sys.modules.setdefault("psycopg", fake_psycopg)

fake_psycopg_pool = types.ModuleType("psycopg_pool")
fake_psycopg_pool.ConnectionPool = object
sys.modules.setdefault("psycopg_pool", fake_psycopg_pool)

# Ensure required environment variables exist before importing the worker module
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
os.environ.setdefault("RERANK_MODEL", "test-rerank-model")

import polars.app.rerank_worker as rerank_worker


def test_cosine_similarity_basic():
    vec = [1.0, 2.0, 3.0]
    assert rerank_worker.cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_handles_zero_vector():
    assert rerank_worker.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_similarity_mismatched_dimensions():
    assert rerank_worker.cosine_similarity([1.0, 0.0], [1.0]) == 0.0


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummyClient:
    def __init__(self, payload):
        self._payload = payload
        self.requests = []

    def post(self, url, json):
        self.requests.append((url, json))
        return DummyResponse(self._payload)


def test_embed_batch_returns_embeddings():
    payload = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    client = DummyClient(payload)
    inputs = ["query", "chunk"]

    result = rerank_worker.embed_batch(client, inputs)

    assert result == payload["embeddings"]
    assert client.requests == [
        (f"{rerank_worker.OLLAMA}/api/embed", {"model": rerank_worker.MODEL, "input": inputs})
    ]


def test_embed_batch_handles_single_embedding_format():
    payload = {"embedding": [0.5, 0.6]}
    client = DummyClient(payload)

    result = rerank_worker.embed_batch(client, ["only-item"])

    assert result == [payload["embedding"]]


def test_embed_batch_empty_inputs_returns_empty_list():
    client = DummyClient({})
    assert rerank_worker.embed_batch(client, []) == []
    assert client.requests == []


def test_update_scores_batches_and_commits():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value

    rerank_worker.update_scores(conn, [1, 2], [0.1, 0.2])

    cursor.executemany.assert_called_once()
    args, _ = cursor.executemany.call_args
    assert args[0] == "UPDATE embeddings SET rank_score = %s WHERE chunk_id = %s"
    assert list(args[1]) == [(0.1, 1), (0.2, 2)]
    conn.commit.assert_called_once()


def test_fetch_candidates_executes_expected_query():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    expected_rows = [(1, "chunk text", "title")]
    cursor.fetchall.return_value = expected_rows

    rows = rerank_worker.fetch_candidates(conn, limit=5)

    cursor.execute.assert_called_once()
    executed_query, params = cursor.execute.call_args[0]
    assert "FROM chunks c" in executed_query
    assert params == (5,)
    assert rows == expected_rows
