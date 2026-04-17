import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Optional
from urllib.parse import urlparse
from uuid import uuid4

import requests

from .base_tool import BaseTool
from .schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema
from .schemas import OpenAIFunctionToolSchema, ToolResponse

DEFAULT_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global rate limiter for Jina Reader (browse_webpage).
# Jina free tier allows 20 requests per second.  We use an asyncio.Semaphore
# to cap concurrency and a token-bucket style pacer so that bursts of
# concurrent agent loops don't exceed the limit.
# ---------------------------------------------------------------------------
_JINA_RPM_LIMIT = int(os.getenv("JINA_RATE_LIMIT_RPM", "1200"))  # 20 rps × 60
_JINA_RPS_LIMIT = int(os.getenv("JINA_RATE_LIMIT_RPS", "20"))
_JINA_MAX_RETRIES = int(os.getenv("JINA_MAX_RETRIES", "3"))
_JINA_RETRY_BASE_DELAY = float(os.getenv("JINA_RETRY_BASE_DELAY", "2.0"))


class _AsyncRateLimiter:
    """Simple token-bucket rate limiter for async code.

    Allows up to ``rate`` calls per second.  Each ``acquire()`` waits until a
    token is available.
    """

    def __init__(self, rate: float):
        self._rate = rate
        self._interval = 1.0 / rate
        self._lock = asyncio.Lock()
        self._last: float = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


# Lazily initialised per-event-loop; safe across Ray workers because each
# worker runs its own event loop.
_jina_rate_limiter: _AsyncRateLimiter | None = None


def _get_jina_rate_limiter() -> _AsyncRateLimiter:
    global _jina_rate_limiter
    if _jina_rate_limiter is None:
        _jina_rate_limiter = _AsyncRateLimiter(_JINA_RPS_LIMIT)
    return _jina_rate_limiter


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is not set")
    return value


# ---------------------------------------------------------------------------
# Snippet-based smart truncation utilities (ported from DR-Tulu
# agent/dr_agent/tool_interface/utils.py).
# ---------------------------------------------------------------------------

def _remove_punctuation(text: str) -> str:
    """Remove punctuation from text for better fuzzy matching."""
    return re.sub(r"[^\w\s]", " ", text)


def _f1_score(set1: set, set2: set) -> float:
    """Calculate F1 score between two sets of words."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    if intersection == 0:
        return 0.0
    precision = intersection / len(set1)
    recall = intersection / len(set2)
    return 2 * precision * recall / (precision + recall)


def _sent_tokenize(text: str) -> list[str]:
    """Simple sentence tokenizer (no NLTK dependency)."""
    return re.split(r"(?<=[.!?。！？])\s+", text)


def _extract_snippet_with_context(
    full_text: str, snippet: str, context_chars: int = 3000
) -> tuple[bool, str]:
    """Locate *snippet* inside *full_text* by fuzzy sentence-level matching,
    then return surrounding context of *context_chars* characters on each side.

    This replicates the original DR-Tulu ``extract_snippet_with_context``
    function so that ``browse_webpage`` returns the **most relevant** portion
    of a page rather than a blind head-truncation.

    Returns ``(True, context)`` on success or ``(False, head_fallback)``
    when no sentence scores above the F1 threshold.
    """
    try:
        # Cap input to avoid excessive processing on very large pages.
        full_text = full_text[:100000]

        snippet_lower = _remove_punctuation(snippet.lower())
        snippet_words = set(snippet_lower.split())

        best_sentence: str | None = None
        best_f1 = 0.2  # minimum threshold

        sentences = _sent_tokenize(full_text)

        for sentence in sentences:
            key_sentence = _remove_punctuation(sentence.lower())
            sentence_words = set(key_sentence.split())
            f1 = _f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # Fallback: return the first portion of the full text.
            return False, full_text[: context_chars * 2]
    except Exception:
        return False, full_text[: context_chars * 2]


def _find_snippet_for_url(url: str, messages: list[dict[str, Any]]) -> str:
    """Search the conversation history for the google_search snippet
    associated with *url*.

    The agent loop stores tool responses as messages with ``role='tool'``
    whose content contains ``<snippet id=...>`` blocks.  Each block has a
    ``URL:`` line and a ``Search Snippet:`` line.  We iterate in **reverse**
    order (most recent search first) so that if the same URL appeared in
    multiple searches we prefer the latest snippet.
    """
    # Normalise the target URL for comparison (strip trailing slash,
    # lowercase scheme+host).
    target = url.lower().rstrip("/")

    for msg in reversed(messages):
        text = ""
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if "<snippet " not in text:
            continue
        # Parse individual <snippet> blocks.
        for m in re.finditer(
            r"<snippet\s+id=[^>]*>(.*?)</snippet>", text, re.DOTALL
        ):
            block = m.group(1)
            url_match = re.search(r"URL:\s*(.*)", block)
            snippet_match = re.search(r"Search Snippet:\s*(.*)", block)
            if url_match and snippet_match:
                block_url = url_match.group(1).strip().lower().rstrip("/")
                if block_url == target:
                    return snippet_match.group(1).strip()
    return ""


class _DrTuluBaseTool(BaseTool):
    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema)
        self.timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"call_index": 0}
        return instance_id, ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)

    def _next_result_id(self, instance_id: str, prefix: str) -> str:
        state = self._instance_dict.setdefault(instance_id, {"call_index": 0})
        state["call_index"] += 1
        return f"{prefix}_{instance_id[:8]}_{state['call_index']}"

    @staticmethod
    def _wrap_tool_output(content_blocks: list[str]) -> str:
        content = "\n".join(block for block in content_blocks if block)
        return f"<tool_output>\n{content}\n</tool_output>"

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _get_runway_url(self) -> str:
        return self._clean_text(
            self.config.get(
                "url",
                os.getenv(
                    "RUNWAY_WEB_SEARCH_API_URL",
                    "https://runway.devops.xiaohongshu.com/openai/zhipu/paas/v4/web_search",
                ),
            )
        )

    def _get_runway_api_key(self) -> str:
        api_key_env = self._clean_text(self.config.get("api_key_env")) or "RUNWAY_WEB_SEARCH_API_KEY"
        return _require_env_var(api_key_env)

    @staticmethod
    def _get_nested_value(payload: Any, path: tuple[str, ...]) -> Any:
        current = payload
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    @classmethod
    def _first_text_value(cls, payload: Any, paths: list[tuple[str, ...]]) -> str:
        for path in paths:
            value = cls._get_nested_value(payload, path)
            text = cls._clean_text(value)
            if text:
                return text
        return ""

    @classmethod
    def _score_result_record(cls, record: dict[str, Any]) -> int:
        score = 0
        if cls._clean_text(record.get("title") or record.get("name")):
            score += 3
        if cls._clean_text(record.get("url") or record.get("link") or record.get("href")):
            score += 3
        if cls._clean_text(
            record.get("snippet")
            or record.get("description")
            or record.get("summary")
            or record.get("text")
            or record.get("content")
            or record.get("body")
        ):
            score += 2
        if cls._clean_text(
            record.get("date")
            or record.get("publishedDate")
            or record.get("time")
            or record.get("publish_time")
        ):
            score += 1
        return score

    @classmethod
    def _preferred_candidate_bonus(cls, path: tuple[str, ...]) -> int:
        bonuses = {
            "organic": 6,
            "results": 5,
            "items": 4,
            "documents": 3,
            "docs": 3,
            "pages": 2,
            "value": 1,
            "data": 0,
        }
        return sum(bonuses.get(segment, 0) for segment in path)

    @classmethod
    def _collect_dict_list_candidates(
        cls, payload: Any, path: tuple[str, ...] = ()
    ) -> list[tuple[tuple[str, ...], list[dict[str, Any]]]]:
        candidates: list[tuple[tuple[str, ...], list[dict[str, Any]]]] = []
        if isinstance(payload, list):
            dict_items = [item for item in payload if isinstance(item, dict)]
            if dict_items:
                candidates.append((path, dict_items))
            for index, item in enumerate(payload):
                candidates.extend(cls._collect_dict_list_candidates(item, path + (str(index),)))
            return candidates

        if isinstance(payload, dict):
            preferred_keys = ["organic", "results", "items", "data", "documents", "docs", "value", "pages"]
            iter_keys = preferred_keys + [key for key in payload.keys() if key not in preferred_keys]
            for key in iter_keys:
                if key in payload:
                    candidates.extend(cls._collect_dict_list_candidates(payload[key], path + (key,)))
        return candidates

    @classmethod
    def _find_list_of_dicts(cls, payload: Any) -> list[dict[str, Any]]:
        candidates = cls._collect_dict_list_candidates(payload)
        best_records: list[dict[str, Any]] = []
        best_score: tuple[float, int, int] | None = None

        for path, records in candidates:
            record_scores = [cls._score_result_record(record) for record in records]
            total_score = sum(record_scores)
            max_score = max(record_scores, default=0)
            bonus = cls._preferred_candidate_bonus(path)
            candidate_score = (max_score, total_score + bonus, len(records))
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_records = records

        if best_score is None or best_score[0] <= 0:
            return []
        return best_records

    @classmethod
    def _extract_result_records(cls, payload: Any) -> list[dict[str, Any]]:
        records = cls._find_list_of_dicts(payload)
        if records:
            return records
        if isinstance(payload, dict):
            likely_record = {
                "title": cls._first_text_value(payload, [("title",), ("name",)]),
                "url": cls._first_text_value(payload, [("url",), ("link",), ("href",)]),
                "snippet": cls._first_text_value(
                    payload,
                    [
                        ("snippet",),
                        ("description",),
                        ("summary",),
                        ("text",),
                        ("content",),
                        ("body",),
                    ],
                ),
            }
            if any(likely_record.values()):
                return [likely_record]
        return []

    @classmethod
    def _extract_webpage_payload(cls, payload: Any, fallback_url: str) -> dict[str, str]:
        title = cls._first_text_value(
            payload,
            [
                ("title",),
                ("data", "title"),
                ("metadata", "title"),
                ("result", "title"),
            ],
        )
        page_url = cls._first_text_value(
            payload,
            [
                ("url",),
                ("data", "url"),
                ("metadata", "url"),
                ("result", "url"),
            ],
        ) or fallback_url
        content = cls._first_text_value(
            payload,
            [
                ("content",),
                ("text",),
                ("markdown",),
                ("body",),
                ("result",),
                ("data", "content"),
                ("data", "text"),
                ("data", "markdown"),
                ("data", "body"),
                ("data", "result"),
            ],
        )

        if not content:
            records = cls._extract_result_records(payload)
            if records:
                joined = []
                for record in records[:3]:
                    lines = []
                    title_text = cls._clean_text(record.get("title") or record.get("name"))
                    url_text = cls._clean_text(record.get("url") or record.get("link") or record.get("href"))
                    content_text = cls._clean_text(
                        record.get("content")
                        or record.get("text")
                        or record.get("snippet")
                        or record.get("description")
                        or record.get("summary")
                    )
                    if title_text:
                        lines.append(f"Title: {title_text}")
                    if url_text:
                        lines.append(f"URL: {url_text}")
                    if content_text:
                        lines.append(f"Content: {content_text}")
                    if lines:
                        joined.append("\n".join(lines))
                content = "\n\n".join(joined)

        if not content:
            content = cls._clean_text(payload)

        return {
            "url": page_url,
            "title": title,
            "content": content,
        }

    def _runway_request(self, payload: dict[str, Any]) -> Any:
        response = requests.post(
            self._get_runway_url(),
            headers={
                "api-key": self._get_runway_api_key(),
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _get_runway_search_engine(self, default: str) -> str:
        return self._clean_text(self.config.get("search_engine")) or default

    @classmethod
    def _normalize_url(cls, url: Any) -> str:
        normalized = cls._clean_text(url)
        if not normalized:
            return ""
        if "://" not in normalized:
            normalized = f"https://{normalized.lstrip('/')}"
        parsed = urlparse(normalized)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {normalized}")
        return normalized


class GoogleSearchTool(_DrTuluBaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        properties = {
            "query": OpenAIFunctionPropertySchema(
                type="string",
                description="The search query.",
            ),
        }

        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="google_search",
                description="General web search for relevant webpages and snippets.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties=properties,
                    required=["query"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = self._clean_text(parameters.get("query"))
        if not query:
            return ToolResponse(text="Error: query is required."), 0.0, {"status": "error"}

        # Extract optional search parameters from XML attributes, consistent
        # with the original DR-Tulu / Serper API interface where the model
        # controls geolocation (gl), host language (hl), and result count (num).
        gl = self._clean_text(parameters.get("gl")) or "us"
        hl = self._clean_text(parameters.get("hl")) or "en"
        max_results = int(self.config.get("max_results", 10))
        num_requested = parameters.get("num")
        if num_requested is not None:
            try:
                num_requested = int(num_requested)
                if 1 <= num_requested <= max_results:
                    max_results = num_requested
            except (ValueError, TypeError):
                pass

        def _search():
            search_engine = self._get_runway_search_engine("search_prime")
            if search_engine != "search_prime":
                raise ValueError(
                    f"google_search must use search_engine='search_prime', got '{search_engine}'"
                )
            request_payload = {
                "search_engine": search_engine,
                "search_query": query,
                "query_rewrite": str(self.config.get("query_rewrite", "false")).lower(),
                "gl": gl,
                "hl": hl,
                "num": max_results,
            }
            payload = self._runway_request(request_payload)
            return payload

        try:
            payload = await asyncio.to_thread(_search)
            organic_results = self._extract_result_records(payload)
            # Client-side clipping as a safety net in case the backend
            # returns more results than requested.
            organic_results = organic_results[:max_results]
            blocks = []
            for result in organic_results:
                result_id = self._next_result_id(instance_id, "snippet")
                lines = [
                    f"Title: {self._clean_text(result.get('title') or result.get('name'))}",
                    f"URL: {self._clean_text(result.get('link') or result.get('url') or result.get('href'))}",
                    (
                        "Search Snippet: "
                        f"{self._clean_text(result.get('snippet') or result.get('description') or result.get('summary') or result.get('text') or result.get('content'))}"
                    ),
                ]
                result_date = self._clean_text(
                    result.get("date")
                    or result.get("publishedDate")
                    or result.get("time")
                    or result.get("publish_time")
                )
                if result_date:
                    lines.append(f"Date: {result_date}")
                blocks.append(f"<snippet id={result_id}>\n" + "\n".join(lines) + "\n</snippet>")
            if not blocks:
                empty_id = self._next_result_id(instance_id, "snippet")
                blocks.append(f"<snippet id={empty_id}>\nNo search results found.\n</snippet>")
            return (
                ToolResponse(text=self._wrap_tool_output(blocks)),
                0.0,
                {"status": "success", "total_results": len(organic_results), "query_count": 1, "backend": "runway"},
            )
        except Exception as e:
            return ToolResponse(text=f"Error performing google_search: {e}"), 0.0, {"status": "error", "error": str(e)}


class BrowseWebpageTool(_DrTuluBaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="browse_webpage",
                description="Open a specific URL and extract readable page text.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "url": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The webpage URL to browse.",
                        ),
                    },
                    required=["url"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        url = self._clean_text(parameters.get("url"))
        if not url:
            return ToolResponse(text="Error: url is required."), 0.0, {"status": "error"}

        normalized_url = self._normalize_url(url)

        # ------------------------------------------------------------------
        # Try to find the search snippet for this URL from conversation
        # history so we can do smart truncation (locate the most relevant
        # region in the page) instead of a blind head-truncation.
        # ------------------------------------------------------------------
        agent_data = kwargs.get("agent_data")
        search_snippet = ""
        if agent_data is not None and hasattr(agent_data, "messages"):
            search_snippet = _find_snippet_for_url(normalized_url, agent_data.messages)

        def _browse() -> dict[str, str]:
            response = requests.get(
                f"https://r.jina.ai/{normalized_url}",
                timeout=self.timeout,
            )
            response.raise_for_status()

            content_type = self._clean_text(response.headers.get("Content-Type")).lower()
            if "json" in content_type:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    extracted = self._extract_webpage_payload(payload.get("data", payload), fallback_url=normalized_url)
                    if self._clean_text(extracted.get("content")):
                        return extracted

            text_content = self._clean_text(response.text)
            if not text_content:
                raise ValueError("Jina Reader response did not include readable page content")
            return {
                "url": normalized_url,
                "title": "",
                "content": text_content,
            }

        try:
            # Rate-limit Jina Reader requests with retry on 429
            limiter = _get_jina_rate_limiter()
            last_error: Exception | None = None
            for attempt in range(_JINA_MAX_RETRIES):
                await limiter.acquire()
                try:
                    payload = await asyncio.to_thread(_browse)
                    break  # success
                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code == 429:
                        last_error = e
                        delay = _JINA_RETRY_BASE_DELAY * (2 ** attempt)
                        logger.info(
                            "Jina 429 rate-limited (attempt %d/%d), retrying in %.1fs: %s",
                            attempt + 1, _JINA_MAX_RETRIES, delay, normalized_url,
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
            else:
                # All retries exhausted
                raise last_error  # type: ignore[misc]

            result_id = self._next_result_id(instance_id, "webpage")
            max_content_length = int(self.config.get("max_content_length", 6000))
            content = self._clean_text(payload.get("content")) or "No content available"

            # --- Smart truncation: locate snippet context, then cap length ---
            if len(content) > max_content_length:
                if search_snippet:
                    # Use half of max_content_length as the context on each
                    # side of the matched sentence, matching DR-Tulu's
                    # ``browse_context_char_length`` semantics.
                    context_chars = max_content_length // 2
                    success, localized = _extract_snippet_with_context(
                        content, search_snippet, context_chars=context_chars
                    )
                    if success:
                        content = localized
                    else:
                        # Fallback returned first 2*context_chars chars;
                        # still cap to max_content_length.
                        content = localized[:max_content_length]
                else:
                    # No snippet available — head-truncation fallback.
                    content = content[:max_content_length]

                if len(content) > max_content_length:
                    content = content[:max_content_length]
                content += "...(truncated)"

            lines = [
                f"Title: {self._clean_text(payload.get('title')) or 'No title available'}",
                f"URL: {self._clean_text(payload.get('url')) or normalized_url}",
                f"Content: {content}",
            ]
            text = self._wrap_tool_output([f"<webpage id={result_id}>\n" + "\n".join(lines) + "\n</webpage>"])
            return ToolResponse(text=text), 0.0, {
                "status": "success",
                "query_count": 1,
                "backend": "jina",
            }
        except Exception as e:
            return ToolResponse(text=f"Error performing browse_webpage: {e}"), 0.0, {"status": "error", "error": str(e)}


