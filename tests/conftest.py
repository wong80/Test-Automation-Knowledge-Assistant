"""Shared pytest fixtures."""

from datetime import datetime, timezone

import pytest
import numpy as np
from minsearch import Index
from sentence_transformers import SentenceTransformer


BASE_CONV = {
    "id": "test-conv",
    "question": "What is FastAPI?",
    "answer": "FastAPI is a web framework.",
    "model_used": "gpt-4o-mini",
    "response_time": 1.5,
    "relevance": "RELEVANT",
    "prompt_tokens": 50,
    "completion_tokens": 30,
    "total_tokens": 80,
    "eval_prompt_tokens": 10,
    "eval_completion_tokens": 5,
    "eval_total_tokens": 15,
    "openai_cost": 0.001,
    "timestamp": datetime.now(timezone.utc),
}


@pytest.fixture
def sample_sitemap_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://fastapi.tiangolo.com/</loc></url>
  <url><loc>https://fastapi.tiangolo.com/tutorial/</loc></url>
  <url><loc>https://fastapi.tiangolo.com/tutorial/path-operation/</loc></url>
  <url><loc>https://fastapi.tiangolo.com/tutorial/body/</loc></url>
  <url><loc>https://fastapi.tiangolo.com/img/favicon.ico</loc></url>
</urlset>"""


@pytest.fixture
def sample_doc_html() -> str:
    return """<html><body>
<div class="content">
  <h1>Tutorial - User Guide</h1>
  <p>Welcome to the FastAPI tutorial.</p>
  <h2>Path Operation</h2>
  <p>A path operation is a combination of an HTTP method and a URL path.</p>
  <p>You define it using decorators like @app.get().</p>
  <h3>Path Operation Example</h3>
  <pre><code>@app.get("/items/{item_id}")</code></pre>
  <p>This example shows a basic path operation.</p>
  <h2>Request Body</h2>
  <p>When you need to send data from a client to your API...</p>
</div></body></html>"""


@pytest.fixture
def sample_multi_heading_html() -> str:
    return """<html><body>
<div class="content">
  <h1>Header 1</h1><p>Content under h1.</p>
  <h2>Header 2</h2><p>Content under h2.</p>
  <h3>Header 3</h3><p>Content under h3.</p>
  <h2>Another h2</h2><p>More content.</p>
</div></body></html>"""


@pytest.fixture
def sample_empty_html() -> str:
    return "<html><body></body></html>"


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "id": "fastapi-path-operation-000",
            "title": "Path Operation",
            "section": "Tutorial - User Guide",
            "content": "A path operation is a combination of an HTTP method and a URL path. "
                       "You define it using decorators like @app.get().",
            "url": "https://fastapi.tiangolo.com/tutorial/path-operation/#path-operation",
            "doc_library": "fastapi",
        },
        {
            "id": "fastapi-request-body-001",
            "title": "Request Body",
            "section": "Tutorial - User Guide",
            "content": "When you need to send data from a client to your API, you declare a model.",
            "url": "https://fastapi.tiangolo.com/tutorial/body/#request-body",
            "doc_library": "fastapi",
        },
        {
            "id": "fastapi-query-params-002",
            "title": "Query Parameters",
            "section": "Tutorial - User Guide",
            "content": "When you declare other function parameters that are not part of the path, "
                       "they are automatically interpreted as query parameters.",
            "url": "https://fastapi.tiangolo.com/tutorial/query-params/#query-parameters",
            "doc_library": "fastapi",
        },
    ]


@pytest.fixture
def sample_ground_truth() -> list[dict]:
    return [
        {"question": "How do I create a path operation in FastAPI?", "relevant_chunk_id": "fastapi-path-operation-000"},
        {"question": "How do I send data in a request body?", "relevant_chunk_id": "fastapi-request-body-001"},
        {"question": "How do query parameters work?", "relevant_chunk_id": "fastapi-query-params-002"},
    ]


@pytest.fixture
def sample_index(sample_chunks) -> Index:
    index = Index(text_fields=["title", "section", "content"], keyword_fields=["id", "doc_library"])
    index.fit(sample_chunks)
    return index


@pytest.fixture(scope="session")
def embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def sample_embeddings(embedding_model, sample_chunks) -> np.ndarray:
    texts = [c["content"] for c in sample_chunks]
    return embedding_model.encode(texts, convert_to_numpy=True)


@pytest.fixture
def mock_openai_response(mocker):
    """Mock app.llm.completion (litellm) with a mutable response object.

    Returned mock mirrors the litellm ModelResponse shape: llm() reads
    choices[0].message.content and usage.{prompt,completion,total}_tokens.
    Tests may override the JSON by assigning fixture.output_text.
    """
    import app.llm
    import json
    import re

    response = mocker.MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    response.output_text = json.dumps({
        "relevance": "RELEVANT", "question": "What is a path operation?",
        "relevant_chunk_id": "test-001", "reason": "Test reason.",
    })
    response.choices[0].message.content = response.output_text

    def _make_response(**kw):
        content = (kw.get("messages") or [{}])[0].get("content", "")
        chunk_id = "test-001"
        m = re.search(r'relevant_chunk_id": "([^"]+)"', content)
        if m:
            chunk_id = m.group(1)
        m = re.search(r'Title: (.+)', content)
        title = m.group(1).strip() if m else "Section"

        current = json.loads(response.output_text) if isinstance(response.output_text, str) else {}
        current["relevance"] = current.get("relevance", "RELEVANT")
        current["question"] = f"What is {title}?"
        current["relevant_chunk_id"] = chunk_id
        current.setdefault("reason", "Test reason.")
        response.output_text = json.dumps(current)
        response.choices[0].message.content = response.output_text
        return response

    mock_completion = mocker.MagicMock()
    mock_completion.side_effect = _make_response
    mocker.patch("app.llm.completion", mock_completion)
    return response


@pytest.fixture
def db_connection(mocker):
    """Mock psycopg2 connection for unit tests."""
    mock_conn = mocker.MagicMock()
    mock_cursor = mocker.MagicMock()
    mock_cursor.fetchone.return_value = None
    _last_query = [None]
    def _execute(sql, *a, **kw):
        _last_query[0] = sql
    mock_cursor.execute.side_effect = _execute
    columns_by_table = {
        "conversations": [
            ("id", "text"), ("question", "text"), ("answer", "text"),
            ("model_used", "text"), ("response_time", "float"),
            ("relevance", "text"), ("prompt_tokens", "integer"),
            ("completion_tokens", "integer"), ("total_tokens", "integer"),
            ("eval_prompt_tokens", "integer"), ("eval_completion_tokens", "integer"),
            ("eval_total_tokens", "integer"), ("openai_cost", "float"),
            ("timestamp", "timestamp with time zone"),
        ],
        "feedback": [
            ("id", "integer"), ("conversation_id", "text"),
            ("feedback", "integer"), ("timestamp", "timestamp with time zone"),
        ],
    }
    def _fetchall():
        sql = (_last_query[0] or "").lower()
        for table, cols in columns_by_table.items():
            if table in sql:
                return cols
        return []
    mock_cursor.fetchall.side_effect = _fetchall
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.__enter__.return_value = mock_cursor
    return mock_conn


@pytest.fixture
def grafana_api(mocker):
    """Mock httpx client for Grafana API calls."""
    mock = mocker.MagicMock()
    mock.__enter__.return_value = mock
    return mock


