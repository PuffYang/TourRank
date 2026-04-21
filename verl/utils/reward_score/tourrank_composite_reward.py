"""
Composite reward function for TourRank that combines:
1. GPT Judge Normalized Score (weight: 0.5)
2. Citation Reward (weight: 0.2)
3. Format Reward (weight: 0.2)
4. Search Turns Reward (weight: 0.1)

The citation reward follows the DR-Tulu RL citation scoring recipe:
0.6 * average claim-level citation F1 + 0.4 * citation format reward.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import AzureOpenAI

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

REWARD_WEIGHTS = {
    "gpt_judge_normalized_score": 0.5,
    "citation_reward": 0.2,
    "format_reward": 0.2,
    "search_turns_reward": 0.1,
}

_CLIENT_CACHE: dict[tuple[str, str, str], AzureOpenAI] = {}


citation_recall_has_citation_prompt = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether this statement is supported by the snippet. Please use the following scale to generate your rating:
- [[Fully supported]] - Most information in the statement is supported by or extracted from the snippet. This applies only to cases where the statement and parts of the snippet are almost identical.
- [[Partially supported]] - More than half of the content in the statement is supported by the snippet, but a small portion is either not mentioned or contradicts the snippet. For example, if the statement has two key points and the snippet supports only one of them, it should be considered [Partially supported].
- [[No support]] - The statement is largely unrelated to the snippet, or most key points in the statement do not align with the content of the snippet.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<statement>
{statement}
</statement>

<snippet>
{concatenated_cited_snippets}
</snippet>"""


citation_recall_no_citation_prompt = """You are an expert in evaluating text quality. You will receive a user's question regarding their uploaded document (due to the length of the document, it is not shown to you), an AI assistant's response based on the document, and a sentence from the response. Your task is to determine whether this sentence is a factual statement made based on the information in the document that requires citation, rather than an introductory sentence, transition sentence, or a summary, reasoning, or inference based on the previous response.
Ensure that you do not use any other external information during your evaluation.
Please first provide your judgment (answer with [[Yes]] or [[No]]), then provide your analysis in the format "Need Citation: [[Yes/No]] Analysis: ...".

<question>
{question}
</question>

<response>
{full_response}
</response>

<statement>
{statement}
</statement>"""


citation_precision_prompt = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether the snippet contains some key information of the statement. Please use the following grades to generate the rating:
- [[Relevant]] - Some key points of the statement are supported by the snippet or extracted from it.
- [[Unrelevant]] - The statement is almost unrelated to the snippet, or all key points of the statement are inconsistent with the snippet content.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<statement>
{statement}
</statement>

<snippet>
{concatenated_cited_snippets}
</snippet>"""


def extract_answer_context_citations(response: str) -> Tuple[Optional[str], Optional[str], Dict[str, str]]:
    extracted_answer = None
    extracted_citations: Dict[str, str] = {}
    extracted_context = None

    answer_match = re.search(r"<answer>", response)
    if answer_match:
        context_text = response[: answer_match.start()].strip()
        extracted_citations = extract_citations_from_context(context_text)
        extracted_context = context_text
    else:
        extracted_citations = extract_citations_from_context(response)
        extracted_context = response

    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not match:
        logger.warning("No <answer></answer> tags found in response")
        return None, None, {}

    extracted_answer = match.group(1).strip()
    return extracted_context, extracted_answer, extracted_citations


def extract_citations_from_context(context: str) -> Dict[str, str]:
    citations: Dict[str, str] = {}
    pattern1 = r'<snippets?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</snippets?>'
    pattern2 = r'<snippet?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</snippet?>'
    pattern3 = r'<webpage?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</webpage?>'

    matches1 = re.findall(pattern1, context, re.DOTALL)
    matches2 = re.findall(pattern2, context, re.DOTALL)
    matches3 = re.findall(pattern3, context, re.DOTALL)

    for _, snippet_id, search_results in matches1 + matches2 + matches3:
        citations[snippet_id.strip()] = search_results.strip()
    return citations


def extract_search_tool_calls(context: str, mcp_parser_name: Optional[str] = None) -> List[str]:
    if not mcp_parser_name:
        matches = re.findall(r"<search>(.*?)</search>", context, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]
    if mcp_parser_name == "unified":
        matches = re.findall(r"<tool name=[\"']?([^\"'>]+)[\"']?>(.*?)</tool>", context, re.DOTALL)
    elif mcp_parser_name in {"v20250824", "dr_tulu_xml"}:
        matches = re.findall(r"<call_tool name=[\"']?([^\"'>]+)[\"']?>(.*?)</call_tool>", context, re.DOTALL)
        if not matches:
            matches = re.findall(r"<call_tool name=[\"']?([^\"'>]+)[\"']?>(.*?)</call>", context, re.DOTALL)
    else:
        raise ValueError(f"Unsupported MCP parser name: {mcp_parser_name}")

    result = []
    for tool_name, content in matches:
        if tool_name.strip().lower() in ("google_search", "browse_webpage") and content.strip():
            result.append(content.strip())
    return result


def compute_format_reward(response: str, mcp_parser_name: Optional[str] = None) -> float:
    answer_format_reward = 1.0 if re.search(r"<answer>.*?</answer>", response, re.DOTALL) else 0.0
    citation_format_reward = 1.0 if re.search(r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>", response, re.DOTALL) else 0.0
    query_format_reward = 1.0 if extract_search_tool_calls(response, mcp_parser_name=mcp_parser_name) else 0.0
    return 0.5 * answer_format_reward + 0.3 * citation_format_reward + 0.2 * query_format_reward


def compute_search_turns_reward(context: str, upper_bound: int = 3, mcp_parser_name: Optional[str] = None) -> Tuple[float, int]:
    if not context:
        return 0.0, 0
    queries = extract_search_tool_calls(context, mcp_parser_name=mcp_parser_name)
    num_valid_queries = len(queries)
    return min(float(num_valid_queries) / upper_bound, 1.0), num_valid_queries


def extract_claims_and_corresponding_citation_ids(
    response: str,
    split_non_cited_parts_by_newlines: bool = False,
    split_non_cited_parts_by_sentences: bool = False,
) -> Dict[str, List[str]]:
    claims: Dict[str, List[str]] = {}
    cite_pattern = r"<cite id=([\"\']?)([^\"\'>\s]+)\1[^>]*>([^<]+)</cite>"
    cite_matches = re.findall(cite_pattern, response)

    cite_tag_pattern = r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>"
    non_cited_parts = re.split(cite_tag_pattern, response)

    if split_non_cited_parts_by_newlines:
        split_parts: List[str] = []
        for part in non_cited_parts:
            split_parts.extend(re.split(r"\n", part))
        non_cited_parts = split_parts

    if split_non_cited_parts_by_sentences:
        split_parts = []
        for part in non_cited_parts:
            split_parts.extend(re.split(r"[.!?]", part))
        non_cited_parts = split_parts

    for part in non_cited_parts:
        part = part.strip()
        if part:
            claims[part] = []

    for _, citation_ids, cited_text in cite_matches:
        cited_text = cited_text.strip()
        if cited_text:
            claims[cited_text] = [cid.strip() for cid in citation_ids.split(",") if cid.strip()]

    return claims


def score_citation_format(claims: Dict[str, List[str]], citations: Dict[str, str]) -> float:
    all_citations: List[str] = []
    for citation_ids in claims.values():
        all_citations.extend(citation_ids)
    all_citations = list(set(all_citations))
    if not all_citations:
        return 0.0
    valid_citations = [citation for citation in all_citations if citation in citations]
    return len(valid_citations) / len(all_citations)


def _get_citation_judge_config(gpt_judge_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(gpt_judge_config or {})
    api_key = cfg.get("api_key")
    api_version = cfg.get("api_version")
    azure_endpoint = cfg.get("azure_endpoint")

    api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    api_version = api_version or os.environ.get("OPENAI_API_VERSION", "2024-06-01")
    azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")

    return {
        "model": str(cfg.get("citation_judge_model") or cfg.get("model", "gpt-4o-mini")),
        "temperature": float(cfg.get("citation_judge_temperature", 0.0)),
        "max_tokens": int(cfg.get("citation_judge_max_tokens", 800)),
        "timeout": int(cfg.get("citation_judge_timeout", cfg.get("timeout", 200))),
        "max_concurrency": max(1, int(cfg.get("citation_judge_max_concurrency", 8))),
        "api_key": api_key,
        "api_version": str(api_version),
        "azure_endpoint": str(azure_endpoint),
    }


def _get_azure_client(judge_cfg: Dict[str, Any]) -> AzureOpenAI:
    cache_key = (
        judge_cfg["api_key"],
        judge_cfg["api_version"],
        judge_cfg["azure_endpoint"],
    )
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        client = AzureOpenAI(
            api_key=judge_cfg["api_key"],
            api_version=judge_cfg["api_version"],
            azure_endpoint=judge_cfg["azure_endpoint"],
        )
        _CLIENT_CACHE[cache_key] = client
    return client


def _run_citation_judge_sync(user_prompt: str, judge_cfg: Dict[str, Any]) -> str:
    if not judge_cfg["api_key"] or not judge_cfg["azure_endpoint"]:
        logger.warning("Citation judge credentials are missing; returning empty judge output.")
        return ""

    try:
        client = _get_azure_client(judge_cfg)
        response = client.chat.completions.create(
            model=judge_cfg["model"],
            messages=[{"role": "user", "content": user_prompt}],
            temperature=judge_cfg["temperature"],
            max_tokens=judge_cfg["max_tokens"],
            timeout=judge_cfg["timeout"],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("Citation judge request failed: %s", exc)
        return ""


async def _run_citation_judge_async(
    user_prompt: str,
    judge_cfg: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_citation_judge_sync, user_prompt, judge_cfg)


def extract_recall_rating_from_response(response: str) -> float:
    rating = re.search(r"Rating: \[\[(.*)\]\]", response)
    if not rating:
        return 0.0
    extracted_text = rating.group(1).strip().lower()
    if extracted_text == "fully supported":
        return 1.0
    if extracted_text == "partially supported":
        return 0.5
    if extracted_text == "no support":
        return 0.0
    return 0.0


def extract_yes_no_from_response(response: str) -> int:
    yes_no = re.search(r"Need Citation: \[\[(.*)\]\]", response)
    if not yes_no:
        return 0
    extracted_text = yes_no.group(1).strip().lower()
    if extracted_text == "yes":
        return 1
    if extracted_text == "no":
        return 0
    return 0


def extract_relevant_rating_from_response(response: str) -> int:
    rating = re.search(r"Rating: \[\[(.*)\]\]", response)
    if not rating:
        return 0
    extracted_text = rating.group(1).strip().lower()
    if extracted_text == "relevant":
        return 1
    if extracted_text == "unrelevant":
        return 0
    return 0


def score_with_citation_recall(
    question: str,
    claim: str,
    concatenated_citations: str,
    judge_cfg: Dict[str, Any],
) -> float:
    user_prompt = citation_recall_has_citation_prompt.format(
        question=question,
        statement=claim,
        concatenated_cited_snippets=concatenated_citations,
    )
    return extract_recall_rating_from_response(_run_citation_judge_sync(user_prompt, judge_cfg))


def score_no_citation_recall(
    question: str,
    claim: str,
    full_response: str,
    judge_cfg: Dict[str, Any],
) -> float:
    user_prompt = citation_recall_no_citation_prompt.format(
        question=question,
        statement=claim,
        full_response=full_response,
    )
    return 1 - extract_yes_no_from_response(_run_citation_judge_sync(user_prompt, judge_cfg))


def score_citation_recall(
    question: str,
    claim: str,
    concatenated_citations: str,
    full_response: str,
    judge_cfg: Dict[str, Any],
) -> float:
    if not concatenated_citations:
        return score_no_citation_recall(question, claim, full_response, judge_cfg)
    return score_with_citation_recall(question, claim, concatenated_citations, judge_cfg)


def score_citation_precision(
    question: str,
    claim: str,
    concatenated_citations: str,
    judge_cfg: Dict[str, Any],
) -> float:
    if not concatenated_citations:
        return 1.0
    user_prompt = citation_precision_prompt.format(
        question=question,
        statement=claim,
        concatenated_cited_snippets=concatenated_citations,
    )
    return float(extract_relevant_rating_from_response(_run_citation_judge_sync(user_prompt, judge_cfg)))


def score_citation_f1(
    question: str,
    claim: str,
    concatenated_citations: str,
    full_response: str,
    judge_cfg: Dict[str, Any],
) -> float:
    recall = score_citation_recall(question, claim, concatenated_citations, full_response, judge_cfg)
    precision = score_citation_precision(question, claim, concatenated_citations, judge_cfg)
    if recall + precision == 0:
        return 0.0
    return 2 * (recall * precision) / (recall + precision)


def _build_empty_citation_metrics(num_claims: int = 0) -> Dict[str, float]:
    return {
        "citation_reward": 0.0,
        "citation_format_reward": 0.0,
        "citation_avg_claim_recall": 0.0,
        "citation_avg_claim_precision": 0.0,
        "citation_avg_claim_f1": 0.0,
        "citation_num_claims": float(num_claims),
    }


async def score_with_citation_recall_async(
    question: str,
    claim: str,
    concatenated_citations: str,
    judge_cfg: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> float:
    user_prompt = citation_recall_has_citation_prompt.format(
        question=question,
        statement=claim,
        concatenated_cited_snippets=concatenated_citations,
    )
    response = await _run_citation_judge_async(user_prompt, judge_cfg, semaphore)
    return extract_recall_rating_from_response(response)


async def score_no_citation_recall_async(
    question: str,
    claim: str,
    full_response: str,
    judge_cfg: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> float:
    user_prompt = citation_recall_no_citation_prompt.format(
        question=question,
        statement=claim,
        full_response=full_response,
    )
    response = await _run_citation_judge_async(user_prompt, judge_cfg, semaphore)
    return 1 - extract_yes_no_from_response(response)


async def score_citation_recall_async(
    question: str,
    claim: str,
    concatenated_citations: str,
    full_response: str,
    judge_cfg: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> float:
    if not concatenated_citations:
        return await score_no_citation_recall_async(question, claim, full_response, judge_cfg, semaphore)
    return await score_with_citation_recall_async(question, claim, concatenated_citations, judge_cfg, semaphore)


async def score_citation_precision_async(
    question: str,
    claim: str,
    concatenated_citations: str,
    judge_cfg: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> float:
    if not concatenated_citations:
        return 1.0
    user_prompt = citation_precision_prompt.format(
        question=question,
        statement=claim,
        concatenated_cited_snippets=concatenated_citations,
    )
    response = await _run_citation_judge_async(user_prompt, judge_cfg, semaphore)
    return float(extract_relevant_rating_from_response(response))


async def score_citation_f1_async(
    question: str,
    claim: str,
    concatenated_citations: str,
    full_response: str,
    judge_cfg: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> float:
    recall_task = score_citation_recall_async(
        question, claim, concatenated_citations, full_response, judge_cfg, semaphore
    )
    precision_task = score_citation_precision_async(
        question, claim, concatenated_citations, judge_cfg, semaphore
    )
    recall, precision = await asyncio.gather(recall_task, precision_task)
    if recall + precision == 0:
        return 0.0
    return 2 * (recall * precision) / (recall + precision)


def get_in_context_citation_metrics(
    question: str,
    response: str,
    citations: Dict[str, str],
    gpt_judge_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    claims = extract_claims_and_corresponding_citation_ids(response)
    if not citations:
        return _build_empty_citation_metrics(num_claims=len(claims))

    def concatenate_citations(citation_ids: List[str]) -> str:
        if not citation_ids:
            return ""
        return "\n\n".join(citations[citation_id] for citation_id in citation_ids if citation_id in citations)

    judge_cfg = _get_citation_judge_config(gpt_judge_config)
    citation_format_reward = score_citation_format(claims, citations)

    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    for claim_text, citation_ids in claims.items():
        concatenated = concatenate_citations(citation_ids)
        recall = score_citation_recall(
            question,
            claim_text,
            concatenated,
            response,
            judge_cfg,
        )
        precision = score_citation_precision(
            question,
            claim_text,
            concatenated,
            judge_cfg,
        )
        if recall + precision == 0:
            f1 = 0.0
        else:
            f1 = 2 * (recall * precision) / (recall + precision)
        total_recall += recall
        total_precision += precision
        total_f1 += f1
    if claims:
        claim_count = len(claims)
        avg_recall = total_recall / claim_count
        avg_precision = total_precision / claim_count
        avg_f1 = total_f1 / claim_count
    else:
        claim_count = 0
        avg_recall = 0.0
        avg_precision = 0.0
        avg_f1 = 0.0
    return {
        "citation_reward": 0.6 * avg_f1 + 0.4 * citation_format_reward,
        "citation_format_reward": citation_format_reward,
        "citation_avg_claim_recall": avg_recall,
        "citation_avg_claim_precision": avg_precision,
        "citation_avg_claim_f1": avg_f1,
        "citation_num_claims": float(claim_count),
    }


async def get_in_context_citation_metrics_async(
    question: str,
    response: str,
    citations: Dict[str, str],
    gpt_judge_config: Optional[Dict[str, Any]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Dict[str, float]:
    claims = extract_claims_and_corresponding_citation_ids(response)
    if not citations:
        return _build_empty_citation_metrics(num_claims=len(claims))

    def concatenate_citations(citation_ids: List[str]) -> str:
        if not citation_ids:
            return ""
        return "\n\n".join(citations[citation_id] for citation_id in citation_ids if citation_id in citations)

    judge_cfg = _get_citation_judge_config(gpt_judge_config)
    judge_semaphore = semaphore or asyncio.Semaphore(judge_cfg["max_concurrency"])
    citation_format_reward = score_citation_format(claims, citations)

    async def _score_claim(claim_text: str, citation_ids: List[str]) -> Tuple[float, float, float]:
        concatenated = concatenate_citations(citation_ids)
        recall_task = score_citation_recall_async(
            question,
            claim_text,
            response,
            judge_cfg,
            judge_semaphore,
        )
        precision_task = score_citation_precision_async(
            question,
            claim_text,
            concatenated,
            judge_cfg,
            judge_semaphore,
        )
        recall, precision = await asyncio.gather(recall_task, precision_task)
        if recall + precision == 0:
            f1 = 0.0
        else:
            f1 = 2 * (recall * precision) / (recall + precision)
        return recall, precision, f1

    tasks = [_score_claim(claim_text, citation_ids) for claim_text, citation_ids in claims.items()]
    if tasks:
        claim_metrics = await asyncio.gather(*tasks)
        claim_count = len(claim_metrics)
        avg_recall = sum(item[0] for item in claim_metrics) / claim_count
        avg_precision = sum(item[1] for item in claim_metrics) / claim_count
        avg_f1 = sum(item[2] for item in claim_metrics) / claim_count
    else:
        claim_count = 0
        avg_recall = 0.0
        avg_precision = 0.0
        avg_f1 = 0.0
    return {
        "citation_reward": 0.6 * avg_f1 + 0.4 * citation_format_reward,
        "citation_format_reward": citation_format_reward,
        "citation_avg_claim_recall": avg_recall,
        "citation_avg_claim_precision": avg_precision,
        "citation_avg_claim_f1": avg_f1,
        "citation_num_claims": float(claim_count),
    }


def score_in_context_citations(
    question: str,
    response: str,
    citations: Dict[str, str],
    gpt_judge_config: Optional[Dict[str, Any]] = None,
) -> float:
    return get_in_context_citation_metrics(
        question=question,
        response=response,
        citations=citations,
        gpt_judge_config=gpt_judge_config,
    )["citation_reward"]


async def score_in_context_citations_async(
    question: str,
    response: str,
    citations: Dict[str, str],
    gpt_judge_config: Optional[Dict[str, Any]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> float:
    metrics = await get_in_context_citation_metrics_async(
        question=question,
        response=response,
        citations=citations,
        gpt_judge_config=gpt_judge_config,
        semaphore=semaphore,
    )
    return metrics["citation_reward"]


def compute_citation_reward(
    question: str,
    response: str,
    extracted_citations: Dict[str, str],
    gpt_judge_config: Optional[Dict[str, Any]] = None,
) -> float:
    return score_in_context_citations(question, response, extracted_citations, gpt_judge_config=gpt_judge_config)


async def compute_citation_reward_async(
    question: str,
    response: str,
    extracted_citations: Dict[str, str],
    gpt_judge_config: Optional[Dict[str, Any]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> float:
    return await score_in_context_citations_async(
        question,
        response,
        extracted_citations,
        gpt_judge_config=gpt_judge_config,
        semaphore=semaphore,
    )


def normalize_tournament_score(
    tournament_score: float,
    min_score: float = 0.0,
    max_score: float = 10.0,
) -> float:
    if max_score <= min_score:
        return 0.5
    normalized = (float(tournament_score) - min_score) / (max_score - min_score)
    return max(0.0, min(normalized, 1.0))


def compute_composite_reward(
    response: str,
    tournament_cumulative_score: float,
    ground_truth: Optional[Dict[str, Any]] = None,
    mcp_parser_name: Optional[str] = None,
    min_tournament_score: float = 0.0,
    max_tournament_score: float = 10.0,
    gpt_judge_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = {
        "gpt_judge_raw_score": tournament_cumulative_score,
        "gpt_judge_normalized_score": 0.0,
        "citation_reward": 0.0,
        "citation_format_reward": 0.0,
        "citation_avg_claim_recall": 0.0,
        "citation_avg_claim_precision": 0.0,
        "citation_avg_claim_f1": 0.0,
        "citation_num_claims": 0.0,
        "format_reward": 0.0,
        "search_turns_reward": 0.0,
        "num_search_turns": 0,
        "final_reward": 0.0,
        "log_values": {},
    }

    gpt_judge_normalized = normalize_tournament_score(
        tournament_cumulative_score,
        min_score=min_tournament_score,
        max_score=max_tournament_score,
    )
    result["gpt_judge_normalized_score"] = gpt_judge_normalized

    extracted_context, _, extracted_citations = extract_answer_context_citations(response)
    format_reward = compute_format_reward(response, mcp_parser_name=mcp_parser_name)
    result["format_reward"] = format_reward

    search_turns_reward, num_search_turns = compute_search_turns_reward(
        extracted_context or response,
        upper_bound=3,
        mcp_parser_name=mcp_parser_name,
    )
    result["search_turns_reward"] = search_turns_reward
    result["num_search_turns"] = num_search_turns

    citation_metrics = get_in_context_citation_metrics(
        question=ground_truth.get("query", "") if ground_truth else "",
        response=response,
        citations=extracted_citations,
        gpt_judge_config=gpt_judge_config,
    )
    result.update(citation_metrics)
    citation_reward = float(citation_metrics["citation_reward"])

    final_reward = (
        REWARD_WEIGHTS["gpt_judge_normalized_score"] * gpt_judge_normalized
        + REWARD_WEIGHTS["citation_reward"] * citation_reward
        + REWARD_WEIGHTS["format_reward"] * format_reward
        + REWARD_WEIGHTS["search_turns_reward"] * search_turns_reward
    )
    result["final_reward"] = final_reward
    result["log_values"] = {
        "gpt_judge_normalized_score": gpt_judge_normalized,
        "citation_reward": citation_reward,
        "citation_format_reward": result["citation_format_reward"],
        "citation_avg_claim_recall": result["citation_avg_claim_recall"],
        "citation_avg_claim_precision": result["citation_avg_claim_precision"],
        "citation_avg_claim_f1": result["citation_avg_claim_f1"],
        "citation_num_claims": result["citation_num_claims"],
        "format_reward": format_reward,
        "search_turns_reward": search_turns_reward,
        "num_search_turns": num_search_turns,
    }
    return result


async def compute_composite_reward_async(
    response: str,
    tournament_cumulative_score: float,
    ground_truth: Optional[Dict[str, Any]] = None,
    mcp_parser_name: Optional[str] = None,
    min_tournament_score: float = 0.0,
    max_tournament_score: float = 10.0,
    gpt_judge_config: Optional[Dict[str, Any]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Dict[str, Any]:
    result = {
        "gpt_judge_raw_score": tournament_cumulative_score,
        "gpt_judge_normalized_score": 0.0,
        "citation_reward": 0.0,
        "citation_format_reward": 0.0,
        "citation_avg_claim_recall": 0.0,
        "citation_avg_claim_precision": 0.0,
        "citation_avg_claim_f1": 0.0,
        "citation_num_claims": 0.0,
        "format_reward": 0.0,
        "search_turns_reward": 0.0,
        "num_search_turns": 0,
        "final_reward": 0.0,
        "log_values": {},
    }

    gpt_judge_normalized = normalize_tournament_score(
        tournament_cumulative_score,
        min_score=min_tournament_score,
        max_score=max_tournament_score,
    )
    result["gpt_judge_normalized_score"] = gpt_judge_normalized

    extracted_context, _, extracted_citations = extract_answer_context_citations(response)
    format_reward = compute_format_reward(response, mcp_parser_name=mcp_parser_name)
    result["format_reward"] = format_reward

    search_turns_reward, num_search_turns = compute_search_turns_reward(
        extracted_context or response,
        upper_bound=3,
        mcp_parser_name=mcp_parser_name,
    )
    result["search_turns_reward"] = search_turns_reward
    result["num_search_turns"] = num_search_turns

    citation_metrics = await get_in_context_citation_metrics_async(
        question=ground_truth.get("query", "") if ground_truth else "",
        response=response,
        citations=extracted_citations,
        gpt_judge_config=gpt_judge_config,
        semaphore=semaphore,
    )
    result.update(citation_metrics)
    citation_reward = float(citation_metrics["citation_reward"])

    final_reward = (
        REWARD_WEIGHTS["gpt_judge_normalized_score"] * gpt_judge_normalized
        + REWARD_WEIGHTS["citation_reward"] * citation_reward
        + REWARD_WEIGHTS["format_reward"] * format_reward
        + REWARD_WEIGHTS["search_turns_reward"] * search_turns_reward
    )
    result["final_reward"] = final_reward
    result["log_values"] = {
        "gpt_judge_normalized_score": gpt_judge_normalized,
        "citation_reward": citation_reward,
        "citation_format_reward": result["citation_format_reward"],
        "citation_avg_claim_recall": result["citation_avg_claim_recall"],
        "citation_avg_claim_precision": result["citation_avg_claim_precision"],
        "citation_avg_claim_f1": result["citation_avg_claim_f1"],
        "citation_num_claims": result["citation_num_claims"],
        "format_reward": format_reward,
        "search_turns_reward": search_turns_reward,
        "num_search_turns": num_search_turns,
    }
    return result
