import asyncio

import requests

from verl.tools.dr_tulu_search_tools import BrowseWebpageTool, GoogleSearchTool


def test_extract_result_records_prefers_search_like_records():
    payload = {
        "data": [
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "content": "raw"},
        ],
        "organic": [
            {
                "title": "Latest AI News",
                "link": "https://example.com/ai-news",
                "snippet": "Major model launch and policy updates.",
            }
        ],
    }

    records = GoogleSearchTool._extract_result_records(payload)

    assert records == payload["organic"]


def test_extract_result_records_ignores_unrelated_dict_lists():
    payload = {
        "data": [
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "content": "raw"},
        ]
    }

    records = GoogleSearchTool._extract_result_records(payload)

    assert records == []


class _FakeResponse:
    def __init__(self, *, text="", json_payload=None, headers=None):
        self.text = text
        self._json_payload = json_payload
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        if self._json_payload is None:
            raise ValueError("no json payload")
        return self._json_payload


def test_browse_webpage_falls_back_from_runway_to_jina(monkeypatch):
    tool = BrowseWebpageTool(config={"backend": "runway"}, tool_schema=None)
    instance_id, _ = asyncio.run(tool.create("instance-1"))

    def _raise_runway_error(self, payload):
        raise requests.HTTPError("400 Client Error: Bad Request")

    def _fake_jina_get(url, headers=None, timeout=None):
        assert url == "https://r.jina.ai/https://example.com/article"
        return _FakeResponse(
            text="# Example Title\n\nExample article body.",
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )

    monkeypatch.setattr(BrowseWebpageTool, "_runway_request", _raise_runway_error)
    monkeypatch.setattr("verl.tools.dr_tulu_search_tools.requests.get", _fake_jina_get)

    response, _, metrics = asyncio.run(tool.execute(instance_id, {"url": "https://example.com/article"}))

    assert metrics["status"] == "success"
    assert metrics["backend"] == "jina"
    assert metrics["backend_fallback_used"] is True
    assert "<webpage id=webpage_instance-" in response.text
    assert "URL: https://example.com/article" in response.text
    assert "Content: # Example Title" in response.text


def test_browse_webpage_normalizes_url_for_jina(monkeypatch):
    tool = BrowseWebpageTool(config={"backend": "jina"}, tool_schema=None)
    instance_id, _ = asyncio.run(tool.create("instance-2"))

    def _fake_jina_get(url, headers=None, timeout=None):
        assert url == "https://r.jina.ai/https://example.com/path"
        return _FakeResponse(
            json_payload={
                "data": {
                    "url": "https://example.com/path",
                    "title": "Example Page",
                    "content": "Structured page content.",
                }
            },
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr("verl.tools.dr_tulu_search_tools.requests.get", _fake_jina_get)

    response, _, metrics = asyncio.run(tool.execute(instance_id, {"url": "example.com/path"}))

    assert metrics["status"] == "success"
    assert metrics["backend"] == "jina"
    assert "Title: Example Page" in response.text
    assert "URL: https://example.com/path" in response.text
    assert "Content: Structured page content." in response.text
