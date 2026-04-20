"""Tournament-ranking GPT judge reward.

Instead of scoring each rollout independently (absolute scoring), this module
groups rollouts for the same query and runs a multi-round tournament:

1. All *n* rollouts for a query enter the first round as candidates.
2. Each round partitions candidates into groups of fixed size
   ``group_size`` (default 2 for pairwise comparison).  The number of
   groups adapts automatically to the number of alive candidates.
3. For every group a single GPT-4o call compares the candidates and selects
   ``num_winners_per_group`` winners.
4. Winners receive ``score_increment`` points and advance; losers are
   eliminated.
5. Rounds repeat until ``<= target_finalists`` candidates remain.
6. Cumulative tournament scores are normalised to [0, 1] and combined with
   the existing format reward to produce the final reward value.

The module exposes a **batch-level** ``compute_score_batch`` entry-point
(called by the reward manager) so that it can see all rollouts for the
same query simultaneously.  It also keeps a per-item ``compute_score``
that falls back to the original absolute-scoring logic for edge cases
(single rollout per query, etc.).

Concurrency:
- Within a round different groups are judged concurrently (bounded by
  ``max_concurrency``).
- Rounds are sequential because round *k+1* depends on round *k* winners.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from typing import Any

from openai import AzureOpenAI

from verl.utils.reward_score.search_r1_like_qa_em import (
    compute_format_reward as compute_search_r1_format_reward,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

FORMAT_REWARD_WEIGHT = 0.2
_CLIENT_CACHE: dict[tuple[str, str, str], AzureOpenAI] = {}

# ---------------------------------------------------------------------------
# Helpers shared with rubric_gpt_judge (duplicated intentionally so the file
# remains self-contained)
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        return {}


def _get_judge_config(gpt_judge: Any = None, **kwargs) -> dict[str, Any]:
    cfg = {
        "model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 1200,
        "timeout": 200,
        "max_retries": 3,
        "retry_sleep": 1.5,
        "enable_content_filter_retry": True,
        "max_rollout_chars": 12000,
        "fallback_to_first_on_error": True,
        "score_normalization": "fixed_range",
        "score_range_min": 0.0,
        "score_range_max": 10.0,
        "equal_score_reward": 0.5,
        "lose_score": 0.0,
        "api_key": None,
        "api_version": "2024-06-01",
        "azure_endpoint": None,
        # tournament-specific defaults
        "group_size": 2,
        "num_winners_per_group": 1,
        "target_finalists": 1,
        "score_increment": 1.0,
        "max_concurrency": 8,
        "num_tournament_repeats": 1,
    }
    cfg.update(_as_dict(gpt_judge))
    for key in list(cfg):
        if key in kwargs and kwargs[key] is not None:
            cfg[key] = kwargs[key]
    cfg["api_key"] = cfg.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
    cfg["azure_endpoint"] = cfg.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
    cfg["content_filter_fallback_reward"] = cfg.get("content_filter_fallback_reward", cfg["equal_score_reward"])
    return cfg


def _get_client(cfg: dict[str, Any]) -> AzureOpenAI:
    api_key = cfg.get("api_key")
    azure_endpoint = cfg.get("azure_endpoint")
    api_version = str(cfg.get("api_version", "2024-06-01"))
    print(f"[DIAG] _get_client: api_key={'SET' if api_key else 'NONE'}, "
          f"azure_endpoint={azure_endpoint!r}, api_version={api_version}", flush=True)
    if not api_key:
        raise ValueError("Missing Azure OpenAI API key. Set reward.gpt_judge.api_key or AZURE_OPENAI_API_KEY.")
    if not azure_endpoint:
        raise ValueError("Missing Azure OpenAI endpoint. Set reward.gpt_judge.azure_endpoint or AZURE_OPENAI_ENDPOINT.")
    cache_key = (str(api_key), api_version, str(azure_endpoint))
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint)
        _CLIENT_CACHE[cache_key] = client
    return client


def _extract_query_and_rubrics(ground_truth: Any, extra_info: dict[str, Any] | None = None) -> tuple[str, list[dict]]:
    extra_info = extra_info or {}
    if isinstance(ground_truth, dict):
        query = str(ground_truth.get("query", "")).strip()
        rubrics = ground_truth.get("rubrics", [])
    else:
        query = str(extra_info.get("query", "") or ground_truth or "").strip()
        rubrics = []
    if not isinstance(rubrics, list):
        rubrics = []
    return query, rubrics


def _compute_format_reward(text: str, format_penalty: str = "easy") -> dict[str, float] | float:
    mcp_parser_name = "dr_tulu_xml" if "<call_tool name=" in (text or "") else None
    return compute_search_r1_format_reward(
        text or "",
        mcp_parser_name=mcp_parser_name,
        format_penalty=format_penalty,
    )


def _strip_think_blocks(text: str) -> str:
    cleaned = text or ""
    pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
    while True:
        updated = pattern.sub("", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _prepare_response_for_judge(text: str) -> str:
    cleaned = _strip_think_blocks(text)
    answer_blocks = re.findall(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if answer_blocks:
        answer = re.sub(r"</?[^>]+>", "", answer_blocks[-1]).strip()
        if answer:
            return answer

    boxed_answers = re.findall(r"\\boxed\{(.*?)\}", cleaned, flags=re.DOTALL)
    if boxed_answers:
        return boxed_answers[-1].strip()

    meta_prefixes = (
        "decompose the query",
        "assumptions",
        "plan",
        "search plan",
        "goal",
        "next step",
        "sufficiency check",
        "synthesis",
        "proceed to final answer",
        "latest snippets",
        "first result",
        "return the minimal boxed answer",
        "we will provide the minimal direct answer",
    )
    filtered_lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = re.sub(r"^[>\-\*\d\.\)\(\s]+", "", line).strip().lower()
        if not normalized or normalized in {"assistant", "user", "<answer>", "</answer>", "<final_answer>", "</final_answer>"}:
            continue
        if normalized.startswith("final answer") or normalized.startswith("answer:"):
            if ":" in line:
                tail = line.split(":", 1)[1].strip()
                if tail:
                    filtered_lines.append(tail)
            continue
        if any(normalized.startswith(prefix) for prefix in meta_prefixes):
            continue
        filtered_lines.append(line)

    if not filtered_lines:
        return re.sub(r"</?[^>]+>", "", cleaned).strip()

    joined = re.sub(r"</?[^>]+>", "", "\n".join(filtered_lines)).strip()
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", joined) if p.strip()]
    if paragraphs:
        return paragraphs[-1]
    return filtered_lines[-1].strip()


def _sanitize_for_judge(text: str) -> str:
    if not text:
        return text
    cleaned = text
    cleaned = re.sub(r"https?://\S+", "[url]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(sex|sexual|sexy|porn|pornographic|erotic)\b", "[sensitive_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(nude|nudity|naked|explicit)\b", "[sensitive_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(vagina|penis|breast|breasts|genital|genitals)\b", "[sensitive_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(minor|child|children|teen|underage)\b", "[age_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(assault|abuse|abused|harass|harassment)\b", "[sensitive_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(kill|killed|killing|kills)\b", "[violent_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(murder|murdered|murdering)\b", "[violent_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(attack|attacked|attacking|attacks)\b", "[violent_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(shoot|shooting|shot|shooter)\b", "[violent_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(stab|stabbed|stabbing)\b", "[violent_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(bomb|bombed|bombing)\b", "[violent_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(terror|terrorist|terrorism)\b", "[sensitive_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(rape|raped|raping)\b", "[sensitive_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(suicide|self-harm)\b", "[sensitive_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(violent|violence)\b", "[sensitive_term]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(性|色情|裸|裸体|露骨|未成年|儿童|青少年|虐待|骚扰)", "[sensitive_term]", cleaned)
    cleaned = re.sub(r"(杀|谋杀|袭击|枪击|枪杀|刺伤|炸弹|爆炸|恐怖|恐袭|自杀|强奸|暴力|血腥)", "[sensitive_term]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "[empty after policy sanitization]"


def _sanitize_rubrics_for_judge(rubrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_rubrics = []
    for item in rubrics:
        if not isinstance(item, dict):
            continue
        safe_rubrics.append(
            {
                "title": _sanitize_for_judge(str(item.get("title", ""))),
                "description": _sanitize_for_judge(str(item.get("description", item.get("rubric", "")))),
                "weight": item.get("weight", 1),
            }
        )
    return safe_rubrics


def _is_content_filter_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "content_filter" in text or "responsibleaipolicyviolation" in text or "content management policy" in text


def _strip_markdown_json_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Tournament-specific prompt & judging
# ---------------------------------------------------------------------------


def _build_tournament_prompt(
    query: str,
    rubrics: list[dict[str, Any]],
    candidate_texts: list[str],
    num_winners: int,
) -> str:
    """Build a prompt asking GPT to compare multiple candidates and pick winners."""
    rubric_lines = []
    for i, item in enumerate(rubrics):
        title = str(item.get("title", "")).strip()
        description = str(item.get("description", item.get("rubric", ""))).strip()
        weight = item.get("weight", 1)
        rubric_lines.append(f"{i + 1}. title: {title}\n   description: {description}\n   weight: {weight}")
    rubric_block = "\n".join(rubric_lines) if rubric_lines else "No rubrics provided."

    candidate_block_parts = []
    for idx, text in enumerate(candidate_texts):
        candidate_block_parts.append(f"--- Candidate {idx} ---\n{text}\n--- End Candidate {idx} ---")
    candidate_block = "\n\n".join(candidate_block_parts)

    return (
        "You are an expert scorer for deep-research quality.\n"
        "You are given a query, a rubric, and multiple candidate responses.\n"
        "Your task is to compare all candidates against the rubric and select "
        f"the best {num_winners} candidate(s).\n\n"
        "Query:\n"
        f"{query}\n\n"
        "Rubric (title, description, weight):\n"
        f"{rubric_block}\n\n"
        "Candidates:\n"
        f"{candidate_block}\n\n"
        "Instructions:\n"
        f"- Compare all candidates holistically using the rubric.\n"
        f"- Select exactly {num_winners} winner(s) that best satisfy the rubric.\n"
        "- Return ONLY valid JSON (no markdown, no explanation) with this schema:\n"
        '{\n'
        '  "winners": [0, 2],\n'
        '  "reasoning": "brief one-line justification"\n'
        '}\n'
        "- The \"winners\" field must contain the candidate indices (0-based) of the winners.\n"
        f"- You must select exactly {num_winners} winner(s).\n"
    )


def _parse_tournament_output(text: str, num_candidates: int, num_winners: int) -> list[int]:
    """Parse the GPT output to extract winner indices.

    Returns a list of winner indices (0-based). Falls back to random
    selection when parsing fails.
    """
    cleaned = _strip_markdown_json_fence(text)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = None

    winners: list[int] | None = None
    if isinstance(parsed, dict):
        raw_winners = parsed.get("winners", parsed.get("winner", None))
        if isinstance(raw_winners, list):
            winners = [int(w) for w in raw_winners if 0 <= int(w) < num_candidates]
        elif isinstance(raw_winners, (int, float)):
            w = int(raw_winners)
            if 0 <= w < num_candidates:
                winners = [w]

    if winners is None:
        # Try to extract numbers from raw text
        nums = re.findall(r"\b(\d+)\b", cleaned)
        winners = [int(n) for n in nums if 0 <= int(n) < num_candidates]

    # De-duplicate while preserving order
    seen = set()
    unique_winners = []
    for w in winners:
        if w not in seen:
            seen.add(w)
            unique_winners.append(w)
    winners = unique_winners

    # Ensure we have exactly num_winners
    if len(winners) > num_winners:
        winners = winners[:num_winners]
    elif len(winners) < num_winners:
        # Fill with random candidates not already selected
        remaining = [i for i in range(num_candidates) if i not in set(winners)]
        random.shuffle(remaining)
        winners.extend(remaining[: num_winners - len(winners)])

    return winners


def _judge_tournament_group(
    prompt: str,
    cfg: dict[str, Any],
    num_candidates: int,
    num_winners: int,
    fallback_prompt: str | None = None,
    redacted_prompt: str | None = None,
) -> list[int]:
    """Call GPT to judge one group and return winner indices."""
    print(f"[DIAG] _judge_tournament_group: num_candidates={num_candidates}, "
          f"num_winners={num_winners}", flush=True)
    client = _get_client(cfg)
    prompt_candidates = [prompt]
    if fallback_prompt:
        prompt_candidates.append(fallback_prompt)
    if redacted_prompt:
        prompt_candidates.append(redacted_prompt)
    prompt_index = 0

    for attempt in range(1, int(cfg["max_retries"]) + 1):
        try:
            resp = client.chat.completions.create(
                model=str(cfg["model"]),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict deep-research scoring expert. "
                            "Compare the candidate responses and select winners. "
                            "Return only valid JSON without markdown."
                        ),
                    },
                    {"role": "user", "content": prompt_candidates[prompt_index]},
                ],
                temperature=float(cfg["temperature"]),
                max_completion_tokens=int(cfg["max_tokens"]),
                timeout=int(cfg["timeout"]),
            )
            content = resp.choices[0].message.content or ""
            winners = _parse_tournament_output(content, num_candidates, num_winners)
            print(f"[DIAG] _judge_tournament_group: API SUCCESS, winners={winners}", flush=True)
            return winners
        except Exception as exc:
            print(f"[DIAG] _judge_tournament_group: API ERROR attempt {attempt}: {exc!r}", flush=True)
            if _is_content_filter_error(exc) and prompt_index < len(prompt_candidates) - 1:
                prompt_index += 1
                logger.warning(
                    "Tournament judge prompt hit content filter; switch to safer prompt stage %d/%d.",
                    prompt_index + 1,
                    len(prompt_candidates),
                )
                continue
            if attempt < int(cfg["max_retries"]):
                time.sleep(float(cfg["retry_sleep"]) * attempt)

    # All retries exhausted – random fallback
    print(f"[DIAG] _judge_tournament_group: ALL RETRIES EXHAUSTED, random fallback", flush=True)
    logger.warning("Tournament judge failed after %d retries, falling back to random selection.", cfg["max_retries"])
    indices = list(range(num_candidates))
    random.shuffle(indices)
    return indices[:num_winners]


# ---------------------------------------------------------------------------
# Tournament execution (async)
# ---------------------------------------------------------------------------


async def _run_tournament_async(
    query: str,
    rubrics: list[dict[str, Any]],
    rollout_texts: list[str],
    cfg: dict[str, Any],
) -> list[float]:
    """Run the full tournament for one query and return per-rollout scores.

    Parameters
    ----------
    rollout_texts : list[str]
        Prepared response texts for each rollout (already truncated / cleaned).
    cfg : dict
        Judge config including tournament hyper-parameters.

    Returns
    -------
    list[float]
        Cumulative tournament score for each rollout (index-aligned with
        *rollout_texts*).
    """
    n = len(rollout_texts)
    group_size = max(2, int(cfg.get("group_size", 2)))
    num_winners_per_group = max(1, int(cfg.get("num_winners_per_group", 1)))
    target_finalists = max(1, int(cfg.get("target_finalists", 1)))
    score_increment = float(cfg.get("score_increment", 1.0))
    max_concurrency = max(1, int(cfg.get("max_concurrency", 8)))

    # Cumulative scores (indexed by original rollout index)
    scores = [0.0] * n

    # Edge case: only 1 rollout → no comparison needed
    if n <= 1:
        return scores

    # candidate_indices tracks which original indices are still alive
    candidate_indices: list[int] = list(range(n))

    round_num = 0
    semaphore = asyncio.Semaphore(max_concurrency)
    loop = asyncio.get_event_loop()

    while len(candidate_indices) > target_finalists:
        round_num += 1
        num_alive = len(candidate_indices)

        # Determine number of groups from fixed group_size.
        # Use floor division so every group has at least ``group_size``
        # members; the last group may be slightly larger when there is
        # a remainder, avoiding groups of size 1 that would auto-advance
        # without any comparison.
        effective_num_groups = max(1, num_alive // group_size)

        # Partition candidates into groups as evenly as possible
        random.shuffle(candidate_indices)
        groups: list[list[int]] = [[] for _ in range(effective_num_groups)]
        for i, idx in enumerate(candidate_indices):
            groups[i % effective_num_groups].append(idx)

        # Remove empty groups (shouldn't happen, but be safe)
        groups = [g for g in groups if g]

        logger.info(
            "Tournament round %d: %d candidates -> %d groups (group_size=%d)",
            round_num,
            len(candidate_indices),
            len(groups),
            group_size,
        )

        print(
            f"[DIAG] Tournament round {round_num}: {num_alive} candidates -> "
            f"{len(groups)} groups, sizes={[len(g) for g in groups]}",
            flush=True,
        )

        # Judge each group concurrently
        async def _judge_group(group: list[int]) -> list[int]:
            async with semaphore:
                group_texts = [rollout_texts[i] for i in group]
                effective_winners = min(num_winners_per_group, len(group) - 1)
                if effective_winners < 1:
                    # group too small, everyone wins
                    return group

                prompt = _build_tournament_prompt(
                    query=query,
                    rubrics=rubrics,
                    candidate_texts=group_texts,
                    num_winners=effective_winners,
                )

                fallback_prompt = None
                redacted_prompt = None
                if bool(cfg.get("enable_content_filter_retry", True)):
                    safe_query = _sanitize_for_judge(query)
                    safe_rubrics = _sanitize_rubrics_for_judge(rubrics)
                    safe_texts = [_sanitize_for_judge(t) for t in group_texts]
                    fallback_prompt = _build_tournament_prompt(
                        query=safe_query,
                        rubrics=safe_rubrics,
                        candidate_texts=safe_texts,
                        num_winners=effective_winners,
                    )
                    redacted_prompt = _build_tournament_prompt(
                        query="[policy-redacted query]",
                        rubrics=safe_rubrics,
                        candidate_texts=[
                            "[policy-redacted response; evaluate conservatively]"
                        ] * len(group_texts),
                        num_winners=effective_winners,
                    )

                # Run blocking GPT call in executor
                winner_local_indices = await loop.run_in_executor(
                    None,
                    _judge_tournament_group,
                    prompt,
                    cfg,
                    len(group),
                    effective_winners,
                    fallback_prompt,
                    redacted_prompt,
                )

                # Map local winner indices back to original rollout indices
                winner_original_indices = [group[i] for i in winner_local_indices if i < len(group)]
                return winner_original_indices

        tasks = [_judge_group(group) for group in groups]
        results = await asyncio.gather(*tasks)

        # Collect this round's winners
        next_candidates: list[int] = []
        for winner_indices in results:
            for idx in winner_indices:
                scores[idx] += score_increment
            next_candidates.extend(winner_indices)

        # De-duplicate (shouldn't happen, but be safe)
        candidate_indices = list(dict.fromkeys(next_candidates))

        # Safety: if no progress is made, break
        if len(candidate_indices) >= n:
            logger.warning("Tournament made no progress in round %d, breaking.", round_num)
            break

    print(f"[DIAG] _run_tournament_async: DONE, final scores={scores}", flush=True)
    return scores


def _normalize_tournament_scores(scores: list[float], equal_score_reward: float = 0.5) -> list[float]:
    """Normalise tournament scores to [0, 1].

    Parameters
    ----------
    scores : list[float]
        Raw cumulative tournament scores.
    equal_score_reward : float
        Value to assign when all scores are identical (default 0.5).
    """
    if not scores:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    denom = max_s - min_s
    if denom < 1e-12:
        # All scores equal – give everyone the same normalised score
        print(f"[DIAG] _normalize_tournament_scores: ALL EQUAL, scores={scores}, "
              f"equal_score_reward={equal_score_reward}", flush=True)
        return [equal_score_reward] * len(scores)
    norm = [(s - min_s) / denom for s in scores]
    print(f"[DIAG] _normalize_tournament_scores: raw={scores}, norm={norm}", flush=True)
    return norm


# ---------------------------------------------------------------------------
# Per-item format reward + final reward assembly
# ---------------------------------------------------------------------------


def _assemble_final_reward(
    normalized_tournament_score: float,
    solution_str: str,
    format_penalty: str,
) -> dict[str, float]:
    """Combine tournament score with format reward to produce final result dict."""
    format_result = _compute_format_reward(solution_str, format_penalty=format_penalty)

    if format_penalty == "strict":
        format_reward = float(format_result["format_reward"])
        retrieval_reward = float(format_result["retrieval_reward"])
        cite_reward = float(format_result["cite_reward"])
        sum_format_reward = float(format_result["sum_format_reward"])
        effective_format_reward = sum_format_reward
    else:
        format_reward = float(format_result)
        weighted_format_reward = FORMAT_REWARD_WEIGHT * format_reward
        effective_format_reward = weighted_format_reward

    final_reward = float(normalized_tournament_score + effective_format_reward)

    result: dict[str, float] = {
        "score": final_reward,
        "gpt_judge_normalized_score": float(normalized_tournament_score),
        "final_reward": final_reward,
        "tournament_cumulative_score": 0.0,  # will be overwritten by compute_score_batch for multi-rollout
    }

    if format_penalty == "strict":
        result["format_reward"] = format_reward
        result["retrieval_reward"] = retrieval_reward
        result["cite_reward"] = cite_reward
        result["sum_format_reward"] = sum_format_reward
        result["citation_violation"] = float(format_result.get("citation_violation", 0.0))
        result["naked_text"] = float(format_result.get("naked_text", 0.0))
    else:
        result["format_reward"] = format_reward
        result["weighted_format_reward"] = float(weighted_format_reward)

    return result


# ---------------------------------------------------------------------------
# Public API – batch-level entry point
# ---------------------------------------------------------------------------


async def compute_score_batch(
    items: list[dict[str, Any]],
    gpt_judge: dict[str, Any] | None = None,
    reward_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> list[dict[str, float]]:
    """Score a batch of rollouts belonging to the **same query** using tournament ranking.

    When ``num_tournament_repeats > 1`` the full tournament is executed
    multiple times (each time starting from scratch with all *n* rollouts)
    and per-rollout scores are accumulated across repeats before a single
    normalisation step.

    Parameters
    ----------
    items : list[dict]
        Each element has keys ``solution_str``, ``ground_truth``, ``extra_info``,
        ``data_source``.
    gpt_judge : dict, optional
        GPT judge config overrides.
    reward_kwargs : dict, optional
        Extra reward kwargs (e.g. ``format_penalty``).

    Returns
    -------
    list[dict]
        Per-rollout reward dicts (same order as *items*).
    """
    reward_kwargs = _as_dict(reward_kwargs)
    format_penalty = str(kwargs.get("format_penalty", reward_kwargs.get("format_penalty", "easy")))
    cfg = _get_judge_config(gpt_judge=gpt_judge, **kwargs)

    num_repeats = max(1, int(cfg.get("num_tournament_repeats", 1)))
    equal_score_reward = float(cfg.get("equal_score_reward", 0.5))

    print(f"[DIAG] tournament_gpt_judge.compute_score_batch: "
          f"num_items={len(items)}, num_tournament_repeats={num_repeats}, "
          f"gpt_judge_keys={list((gpt_judge or {}).keys())}", flush=True)

    if not items:
        return []

    # Extract query & rubrics from the first item (all share the same query)
    first_gt = items[0].get("ground_truth")
    first_extra = items[0].get("extra_info")
    query, rubrics = _extract_query_and_rubrics(first_gt, extra_info=first_extra)

    # Prepare cleaned response texts
    rollout_texts: list[str] = []
    solution_strs: list[str] = []
    for item in items:
        sol = item.get("solution_str", "")
        solution_strs.append(sol)
        text = _prepare_response_for_judge(sol)
        if len(text) > int(cfg["max_rollout_chars"]):
            text = text[: int(cfg["max_rollout_chars"])]
        rollout_texts.append(text)

    n = len(items)

    # Accumulate tournament scores across repeats
    total_scores = [0.0] * n
    for repeat_idx in range(num_repeats):
        if num_repeats > 1:
            print(f"[DIAG] Tournament repeat {repeat_idx + 1}/{num_repeats} starting "
                  f"(all {n} rollouts re-enter)", flush=True)

        repeat_scores = await _run_tournament_async(
            query=query,
            rubrics=rubrics,
            rollout_texts=rollout_texts,
            cfg=cfg,
        )

        for i in range(n):
            total_scores[i] += repeat_scores[i]

        if num_repeats > 1:
            print(f"[DIAG] Tournament repeat {repeat_idx + 1}/{num_repeats} done, "
                  f"repeat_scores={repeat_scores}, total_scores={total_scores}", flush=True)

    # Normalise accumulated scores to [0, 1]
    norm_scores = _normalize_tournament_scores(total_scores, equal_score_reward=equal_score_reward)

    # Assemble per-item results
    results: list[dict[str, float]] = []
    for i, item in enumerate(items):
        result = _assemble_final_reward(
            normalized_tournament_score=norm_scores[i],
            solution_str=solution_strs[i],
            format_penalty=format_penalty,
        )
        result["tournament_cumulative_score"] = total_scores[i]
        result["tournament_num_repeats"] = float(num_repeats)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Single-item fallback (compatible with the standard verl compute_score API)
# ---------------------------------------------------------------------------


def compute_score(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    gpt_judge: dict[str, Any] | None = None,
    reward_kwargs: dict[str, Any] | None = None,
    format_penalty: str = "easy",
    **kwargs,
) -> dict[str, float]:
    """Single-rollout fallback.

    When only one rollout is available (or the reward manager calls per-item),
    we fall back to giving a neutral tournament score of 0.5 and rely on format
    reward only.  The real tournament comparison happens in
    ``compute_score_batch``.
    """
    reward_kwargs = _as_dict(reward_kwargs)
    format_penalty = str(kwargs.get("format_penalty", reward_kwargs.get("format_penalty", format_penalty)))

    return _assemble_final_reward(
        normalized_tournament_score=0.5,
        solution_str=solution_str,
        format_penalty=format_penalty,
    )


compute_score.NEEDS_REWARD_CONFIG = True
