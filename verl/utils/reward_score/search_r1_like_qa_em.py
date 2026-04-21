# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

from __future__ import annotations

import random
import re
import string
from typing import Dict, Optional, Union


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def extract_search_tool_calls(context: str, mcp_parser_name: Optional[str] = None) -> list[str]:
    if not mcp_parser_name:
        patterns = [
            r"<search>(.*?)</search>",
            r"<call_tool name=(.*?)>(.*?)</call_tool>",
            r"<tool name=(.*?)>(.*?)</tool>",
        ]
        extracted_queries = []
        for pattern in patterns:
            matches = re.findall(pattern, context, re.DOTALL)
            if not matches:
                continue
            if isinstance(matches[0], tuple):
                extracted_queries.extend(match[1].strip() for match in matches if match[1].strip())
            else:
                extracted_queries.extend(match.strip() for match in matches if match.strip())
        return extracted_queries

    if mcp_parser_name in {"unified"}:
        matches = re.findall(r"<tool name=(.*?)>(.*?)</tool>", context, re.DOTALL)
    elif mcp_parser_name in {"v20250824", "dr_tulu_xml"}:
        matches = re.findall(r"<call_tool name=(.*?)>(.*?)</call_tool>", context, re.DOTALL)
        if not matches:
            matches = re.findall(r"<call_tool name=(.*?)>(.*?)</call>", context, re.DOTALL)
    else:
        return []

    return [match[1].strip() for match in matches if match[1].strip()]


def _extract_answer_blocks(response: str) -> list[re.Match[str]]:
    return list(re.finditer(r"<answer>.*?</answer>", response, re.DOTALL))


def _extract_model_generated_text(response: str) -> str:
    """Extract only the model-generated portions from a multi-turn response.

    In a multi-turn agent loop the full decoded response has the structure::

        [MODEL_TEXT] \\nuser\\n [SYSTEM_TOOL_RESPONSE] \\nassistant\\n [MODEL_TEXT] ...

    The ``\\nuser\\n`` / ``\\nassistant`` boundaries come from the chat-template
    role markers (e.g. ``<|im_start|>user`` decoded as ``user``).  Everything
    between ``\\nuser\\n`` and ``\\nassistant`` is system-injected content
    (``<tool_response>...</tool_response>``).  This helper strips those parts and
    returns only what the model itself generated.

    For single-turn responses (no ``\\nuser\\n`` marker) the input is returned
    unchanged.

    Note: the trailing ``\\n`` after ``assistant`` may occasionally be absent
    (e.g. when tokenizer merges it with the next token), so we match
    ``\\nassistant`` with an *optional* trailing newline.
    """
    user_segments = re.split(r"\nuser\n", response)
    model_parts = [user_segments[0]]  # first segment is always model output
    for seg in user_segments[1:]:
        # seg = "<tool_response>...</tool_response>\nassistant\n<think>..."
        # Note: \nassistant may or may not be followed by \n
        assistant_split = re.split(r"\nassistant\n?", seg, maxsplit=1)
        if len(assistant_split) == 2:
            model_parts.append(assistant_split[1])
        # else: malformed turn -- skip the system part safely
    return "".join(model_parts)


# Tools defined in the system prompt.  Any ``<call_tool name="...">`` whose
# name is NOT in this set is considered a hallucinated / fabricated tool.
ALLOWED_TOOL_NAMES: frozenset[str] = frozenset({
    "google_search",
    "browse_webpage",
})


def _has_fabricated_tool(response: str) -> bool:
    """Check if the model used a tool that is not in the allowed list.

    The system prompt defines a fixed set of tools (``google_search``,
    ``browse_webpage``).  During RL exploration the model
    may hallucinate tool names that do not exist (e.g. ``download_file``,
    ``python``, ``search``).  Such calls waste a turn and indicate the model
    is not following the tool protocol, so they should be penalized.

    This function extracts *all* ``<call_tool name="...">`` occurrences from
    the model-generated text (system-injected segments are stripped first)
    and checks whether every tool name belongs to :data:`ALLOWED_TOOL_NAMES`.
    """
    model_text = _extract_model_generated_text(response)

    # Extract all <call_tool name="..."> in model text
    tool_calls = re.findall(r'<call_tool\s+name=["\']?(\w+)', model_text)
    for name in tool_calls:
        if name not in ALLOWED_TOOL_NAMES:
            return True
    return False


def _extract_tool_response_ids(response: str) -> set[str]:
    """Extract all valid snippet/webpage IDs from system-injected tool responses.

    In the multi-turn decoded text, tool responses appear between ``\\nuser\\n``
    and ``\\nassistant\\n`` markers.  They contain XML tags like::

        <snippet id=snippet_50c4735d_1>Title: ...\\nURL: ...\\nSnippet: ...</snippet>
        <webpage id=webpage_a1b2c3d4_1>Title: ...\\nURL: ...\\nContent: ...</webpage>

    This function extracts the IDs from those tags.
    """
    # Split by \nuser\n to find system-injected segments
    user_segments = re.split(r"\nuser\n", response)
    tool_response_ids: set[str] = set()
    for seg in user_segments[1:]:
        # Everything before \nassistant\n is a system-injected tool response
        assistant_split = re.split(r"\nassistant\n?", seg, maxsplit=1)
        system_part = assistant_split[0]
        # Extract <snippet id=XXX> and <webpage id=XXX>
        ids = re.findall(r"<(?:snippet|webpage)\s+id=([^>]+)>", system_part)
        tool_response_ids.update(id_str.strip() for id_str in ids)
    return tool_response_ids


def _extract_cite_ids_from_answer(response: str) -> list[str]:
    """Extract all cite IDs from ``<cite id="...">`` tags in the ``<answer>`` block.

    Returns a list of cite IDs (may contain duplicates).  Returns an empty
    list if no ``<answer>`` block or no ``<cite>`` tags are found.
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not answer_match:
        return []
    answer_text = answer_match.group(1)
    # Match <cite id="X"> or <cite id='X'> or <cite id=X>
    # The ID value may contain commas for multi-cite (e.g. id="id1,id2")
    raw_ids = re.findall(r'<cite\s+id=["\']?([^"\'>\s]+)["\']?[^>]*>', answer_text)
    # Split comma-separated IDs (e.g. "snippet_a_1,snippet_b_2")
    cite_ids: list[str] = []
    for raw_id in raw_ids:
        cite_ids.extend(part.strip() for part in raw_id.split(",") if part.strip())
    return cite_ids


def _has_citation_violation(response: str, used_tools: bool) -> bool:
    """Check citation violations (condition ④).

    Condition ④a: If the model did **not** use tools, ``<answer>`` must **not**
    contain any ``<cite>`` tags (citations without tool evidence are fabricated).

    Condition ④b: If the model used tools **and** has ``<cite>`` tags in
    ``<answer>``, every cite ID must reference a real ID from a tool response
    (``<snippet id=...>`` or ``<webpage id=...>``).  Hallucinated cite IDs are
    a format violation.

    Note: using tools but having **no** ``<cite>`` tags is **allowed** (no
    penalty).  The presence of citations when tools were used is instead
    encouraged via a separate positive ``cite_reward``.

    Returns ``True`` if a violation is detected, ``False`` otherwise.
    """
    cite_ids = _extract_cite_ids_from_answer(response)

    if not used_tools:
        # ④a: no tools used → must NOT have cite tags
        if cite_ids:
            return True
        return False

    # Tools were used.  If no cite tags → OK (no violation; cite_reward handles this)
    if not cite_ids:
        return False

    # ④b: all cite IDs must be valid (present in tool responses)
    valid_ids = _extract_tool_response_ids(response)
    for cid in cite_ids:
        if cid not in valid_ids:
            return True

    return False


# Threshold (in characters) for naked text detection.  Text fragments
# shorter than this after stripping whitespace are tolerated (they are
# typically minor formatting noise such as stray newlines or spaces).
_NAKED_TEXT_THRESHOLD: int = 10


def _has_naked_text(response: str) -> bool:
    """Check if the model produced substantial text outside recognised tags.

    In a well-formed response every piece of model-generated text should be
    inside ``<think>...</think>``, ``<call_tool ...>...</call_tool>``, or
    ``<answer>...</answer>`` tags.  Any *naked* text (e.g. ``Let me search
    for this...``) leaks chain-of-thought reasoning outside of the structured
    format and should be penalised.

    The check operates only on model-generated segments (system-injected
    ``<tool_response>`` parts are stripped first via
    :func:`_extract_model_generated_text`).

    A small tolerance (:data:`_NAKED_TEXT_THRESHOLD` characters) is applied
    so that minor whitespace / formatting noise does not trigger a false
    positive.

    Returns ``True`` if a violation is detected, ``False`` otherwise.
    """
    model_text = _extract_model_generated_text(response)

    # Remove all <think>...</think> blocks (greedy across newlines)
    stripped = re.sub(r"<think>.*?</think>", "", model_text, flags=re.DOTALL)
    # Remove all <call_tool ...>...</call_tool> blocks
    stripped = re.sub(r"<call_tool\b[^>]*>.*?</call_tool>", "", stripped, flags=re.DOTALL)
    # Remove all <answer>...</answer> blocks
    stripped = re.sub(r"<answer>.*?</answer>", "", stripped, flags=re.DOTALL)

    # What remains should be negligible whitespace / formatting noise
    remaining = stripped.strip()
    return len(remaining) > _NAKED_TEXT_THRESHOLD


def _compute_strict_format_reward(
    response: str, mcp_parser_name: Optional[str] = None, use_full_response_as_answer: bool = False
) -> dict[str, float]:
    """Compute strict format reward inspired by R1-Searcher.

    Returns a dict with seven keys:
      - ``format_reward``     : -1.0 or 0.0   (base format correctness)
      - ``retrieval_reward``  : 0.0 or 0.1    (bonus for using tools)
      - ``cite_reward``       : 0.0 or 0.1    (bonus for citing when tools used)
      - ``sum_format_reward`` : format_reward + retrieval_reward + cite_reward
      - ``citation_violation``: 0.0 or 1.0    (diagnostic flag)
      - ``naked_text``        : 0.0 or 1.0    (diagnostic flag)
    """
    if use_full_response_as_answer:
        return {
            "format_reward": -1.0, "retrieval_reward": 0.0,
            "cite_reward": 0.0,
            "sum_format_reward": -1.0, "citation_violation": 0.0,
            "naked_text": 0.0,
        }

    # --- Condition ①: single <answer> block ---
    answer_blocks = _extract_answer_blocks(response)
    open_count, close_count = count_answer_tags(response)
    has_single_answer_block = len(answer_blocks) == 1 and open_count == 1 and close_count == 1

    # --- Condition ②: <answer> is the last top-level block ---
    answer_is_last_top_level_block = False
    if has_single_answer_block:
        answer_block = answer_blocks[0].group(0).strip()
        answer_is_last_top_level_block = response.strip().endswith(answer_block)

    # --- Condition ③: no hallucinated tools ---
    queries = extract_search_tool_calls(response, mcp_parser_name=mcp_parser_name)
    has_fabrication = _has_fabricated_tool(response) if queries else False

    # --- Condition ④: citation constraints ---
    #   ④a: if no tools used, <answer> must NOT have <cite> tags
    #   ④b: if tools used and <cite> tags present, all cite IDs must be real
    #   Note: tools used but no <cite> tags → no violation (cite_reward handles this)
    used_tools = bool(queries)
    citation_violated = _has_citation_violation(response, used_tools)

    # --- Condition ⑤: no naked text outside <think>/<call_tool>/<answer> ---
    has_naked = _has_naked_text(response)

    # --- Format reward: all five conditions must be met ---
    if (has_single_answer_block and answer_is_last_top_level_block
            and not has_fabrication and not citation_violated
            and not has_naked):
        format_reward = 0.0
    else:
        format_reward = -1.0

    # --- Retrieval reward: at least one valid <call_tool> invocation ---
    retrieval_reward = 0.1 if queries else 0.0

    # --- Cite reward: tools used AND <answer> has at least one <cite> tag ---
    cite_reward = 0.0
    if used_tools:
        cite_ids = _extract_cite_ids_from_answer(response)
        if cite_ids:
            cite_reward = 0.1

    return {
        "format_reward": format_reward,
        "retrieval_reward": retrieval_reward,
        "cite_reward": cite_reward,
        "sum_format_reward": format_reward + retrieval_reward + cite_reward,
        "citation_violation": 1.0 if citation_violated else 0.0,
        "naked_text": 1.0 if has_naked else 0.0,
    }


def compute_format_reward(
    response: str,
    mcp_parser_name: Optional[str] = None,
    use_full_response_as_answer: bool = False,
    format_penalty: Optional[str] = None,
) -> dict[str, float]:
    """Compute strict format reward.

    The legacy ``format_penalty`` switch is ignored; only the original strict
    format reward is retained.
    """
    return _compute_strict_format_reward(
        response,
        mcp_parser_name=mcp_parser_name,
        use_full_response_as_answer=use_full_response_as_answer,
    )


def compute_score_em(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                return score
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score


def compute_score(
    solution_str,
    ground_truth,
    method="strict",
    format_score=0.0,
    score=1.0,
    extra_info=None,
    format_penalty=None,
    **kwargs,
):
    extra_info = extra_info or {}
    mcp_parser_name = extra_info.get("mcp_parser_name")
    if mcp_parser_name is None and "<call_tool name=" in solution_str:
        mcp_parser_name = "dr_tulu_xml"

    format_result = compute_format_reward(
        solution_str,
        mcp_parser_name=mcp_parser_name,
    )
    format_reward = format_result["format_reward"]
    retrieval_reward = format_result["retrieval_reward"]
    cite_reward = format_result["cite_reward"]
    sum_format_reward = format_result["sum_format_reward"]
    effective_format_score = sum_format_reward

    answer = extract_solution(solution_str=solution_str)
    is_correct = answer is not None and subem_check(answer, ground_truth["target"])
    reward_score = compute_score_subem(
        solution_str=solution_str,
        ground_truth=ground_truth,
        format_score=effective_format_score,
        method=method,
        score=score,
    )
    result = {
        "score": reward_score,
        "format_reward": format_reward,
        "accuracy_reward": float(is_correct),
    }
    result["retrieval_reward"] = retrieval_reward
    result["cite_reward"] = cite_reward
    result["sum_format_reward"] = sum_format_reward
    result["citation_violation"] = format_result.get("citation_violation", 0.0)
    result["naked_text"] = format_result.get("naked_text", 0.0)
    return result
