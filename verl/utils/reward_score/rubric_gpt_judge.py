from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from openai import AzureOpenAI

from verl.utils.reward_score.search_r1_like_qa_em import compute_format_reward as compute_search_r1_format_reward

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

FORMAT_REWARD_WEIGHT = 0.2
_CLIENT_CACHE: dict[tuple[str, str, str], AzureOpenAI] = {}


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
    """Compute format reward.  Returns a dict for strict mode, float for easy."""
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


def _fallback_score_on_error(exc: Exception, cfg: dict[str, Any]) -> tuple[float, float]:
    score_min = float(cfg["score_range_min"])
    score_max = float(cfg["score_range_max"])
    if _is_content_filter_error(exc):
        normalized = float(cfg["content_filter_fallback_reward"])
        raw = score_min + normalized * (score_max - score_min)
        return raw, normalized
    return score_min, float(cfg["lose_score"])


def _strip_markdown_json_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _parse_judge_output(text: str) -> dict[str, Any]:
    cleaned = _strip_markdown_json_fence(text)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, (int, float)):
        return {"score": float(parsed)}
    try:
        return {"score": float(cleaned)}
    except Exception as exc:
        raise ValueError(f"Judge output is not valid JSON score: {cleaned}") from exc


def _extract_single_score(judge_result: dict[str, Any]) -> float:
    if "score" in judge_result:
        return float(judge_result["score"])
    if "final_score" in judge_result:
        return float(judge_result["final_score"])
    scores = judge_result.get("scores")
    if isinstance(scores, list) and scores:
        return float(scores[0])
    raise ValueError(f"Judge JSON missing score field: {judge_result}")


def _normalize_single_score(raw_score: float, cfg: dict[str, Any]) -> float:
    score_min = float(cfg["score_range_min"])
    score_max = float(cfg["score_range_max"])
    clipped = min(score_max, max(score_min, float(raw_score)))
    denom = score_max - score_min
    if denom <= 1e-12:
        return float(cfg["equal_score_reward"])
    if str(cfg.get("score_normalization", "fixed_range")) not in {"fixed_range", "group_minmax"}:
        logger.warning("Unknown score_normalization=%s, fallback to fixed_range.", cfg.get("score_normalization"))
    return min(1.0, max(0.0, (clipped - score_min) / denom))


def _build_judge_prompt(query: str, rubrics: list[dict[str, Any]], rollout_text: str) -> str:
    rubric_lines = []
    for i, item in enumerate(rubrics):
        title = str(item.get("title", "")).strip()
        description = str(item.get("description", item.get("rubric", ""))).strip()
        weight = item.get("weight", 1)
        rubric_lines.append(f"{i + 1}. title: {title}\n   description: {description}\n   weight: {weight}")
    rubric_block = "\n".join(rubric_lines) if rubric_lines else "No rubrics provided."
    return (
        "You are an expert scorer for deep-research quality.\n"
        "Using the provided rubric, score the response to the query below.\n\n"
        "Query:\n"
        f"{query}\n\n"
        "Response:\n"
        f"{rollout_text}\n\n"
        "Rubric (title, description, weight):\n"
        f"{rubric_block}\n\n"
        "Scoring rule:\n"
        "- Return one final score in [0, 10], where 0 is worst and 10 is best.\n"
        "- Weight indicates relative importance among rubric items, not a normalized coefficient.\n"
        "- Do not output explanations.\n"
        "- Output must be JSON only, with this schema:\n"
        '{\n  "score": 7.5\n}\n'
    )


def _judge_rollout(judge_prompt: str, cfg: dict[str, Any], fallback_judge_prompt: str | None = None, redacted_judge_prompt: str | None = None) -> dict[str, Any]:
    client = _get_client(cfg)
    last_error = ""
    prompt_candidates = [judge_prompt]
    if fallback_judge_prompt:
        prompt_candidates.append(fallback_judge_prompt)
    if redacted_judge_prompt:
        prompt_candidates.append(redacted_judge_prompt)
    prompt_index = 0

    for attempt in range(1, int(cfg["max_retries"]) + 1):
        try:
            resp = client.chat.completions.create(
                model=str(cfg["model"]),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict deep-research scoring expert. Return only valid JSON without markdown.",
                    },
                    {"role": "user", "content": prompt_candidates[prompt_index]},
                ],
                temperature=float(cfg["temperature"]),
                max_completion_tokens=int(cfg["max_tokens"]),
                timeout=int(cfg["timeout"]),
            )
            content = resp.choices[0].message.content or ""
            return _parse_judge_output(content)
        except Exception as exc:
            last_error = str(exc)
            if _is_content_filter_error(exc) and prompt_index < len(prompt_candidates) - 1:
                prompt_index += 1
                logger.warning("Judge prompt hit content filter; switch to safer prompt stage %d/%d.", prompt_index + 1, len(prompt_candidates))
                continue
            if attempt < int(cfg["max_retries"]):
                time.sleep(float(cfg["retry_sleep"]) * attempt)

    raise RuntimeError(last_error)


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
    """Rubric GPT judge reward in the standard verl custom compute_score format."""
    reward_kwargs = _as_dict(reward_kwargs)
    format_penalty = str(kwargs.get("format_penalty", reward_kwargs.get("format_penalty", format_penalty)))
    cfg = _get_judge_config(gpt_judge=gpt_judge, **kwargs)

    query, rubrics = _extract_query_and_rubrics(ground_truth, extra_info=extra_info)
    format_result = _compute_format_reward(solution_str, format_penalty=format_penalty)
    rollout_text = _prepare_response_for_judge(solution_str)
    if len(rollout_text) > int(cfg["max_rollout_chars"]):
        rollout_text = rollout_text[: int(cfg["max_rollout_chars"])]

    judge_prompt = _build_judge_prompt(query=query, rubrics=rubrics, rollout_text=rollout_text)
    fallback_judge_prompt = None
    redacted_judge_prompt = None
    if bool(cfg.get("enable_content_filter_retry", True)):
        safe_query = _sanitize_for_judge(query)
        safe_rollout_text = _sanitize_for_judge(rollout_text)
        safe_rubrics = _sanitize_rubrics_for_judge(rubrics)
        fallback_judge_prompt = _build_judge_prompt(
            query=safe_query,
            rubrics=safe_rubrics,
            rollout_text=safe_rollout_text,
        )
        redacted_judge_prompt = _build_judge_prompt(
            query="[policy-redacted query]",
            rubrics=safe_rubrics,
            rollout_text="[policy-redacted response; evaluate conservatively based on safe visible content only]",
        )

    try:
        judge_result = _judge_rollout(
            judge_prompt=judge_prompt,
            cfg=cfg,
            fallback_judge_prompt=fallback_judge_prompt,
            redacted_judge_prompt=redacted_judge_prompt,
        )
        raw_score = _extract_single_score(judge_result)
        normalized_score = _normalize_single_score(raw_score, cfg)
    except Exception as exc:
        if not bool(cfg.get("fallback_to_first_on_error", True)):
            raise
        raw_score, normalized_score = _fallback_score_on_error(exc, cfg)
        logger.warning("Judge failed, fallback to safe default. error=%s", exc)

    # --- Compute effective format reward and final reward ---
    if format_penalty == "strict":
        # strict mode: format_result is a dict with format_reward, retrieval_reward, cite_reward, sum_format_reward
        format_reward = float(format_result["format_reward"])
        retrieval_reward = float(format_result["retrieval_reward"])
        cite_reward = float(format_result["cite_reward"])
        sum_format_reward = float(format_result["sum_format_reward"])
        effective_format_reward = sum_format_reward
    else:
        # easy mode: format_result is a float, apply weighting
        format_reward = float(format_result)
        weighted_format_reward = FORMAT_REWARD_WEIGHT * format_reward
        effective_format_reward = weighted_format_reward

    final_reward = float(normalized_score + effective_format_reward)

    result = {
        "score": final_reward,
        "gpt_judge_raw_score": float(raw_score),
        "gpt_judge_normalized_score": float(normalized_score),
        "final_reward": final_reward,
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


compute_score.NEEDS_REWARD_CONFIG = True
