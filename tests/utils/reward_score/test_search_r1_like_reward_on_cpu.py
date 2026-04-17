# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from verl.utils.reward_score import search_r1_like_qa_em


# ---------------------------------------------------------------------------
# Easy mode
# ---------------------------------------------------------------------------

def test_compute_format_reward_for_dr_tulu_xml():
    """Easy mode: should return a float score."""
    response = (
        '<think>search first</think>'
        '<call_tool name="google_search">latest ai news</call_tool>'
        '<tool_output><snippet id=s1>Title: A</snippet></tool_output>'
        '<answer><cite id="s1">AI is in the news.</cite></answer>'
    )

    reward = search_r1_like_qa_em.compute_format_reward(response, mcp_parser_name="dr_tulu_xml")

    assert reward == 1.0


def test_compute_score_falls_back_to_format_reward():
    """Easy mode compute_score: format_reward is a float."""
    response = (
        '<think>search first</think>'
        '<call_tool name="google_search">latest ai news</call_tool>'
        '<answer><cite id="s1">wrong answer</cite></answer>'
    )
    ground_truth = {"target": ["correct answer"]}

    result = search_r1_like_qa_em.compute_score(response, ground_truth)

    assert result["accuracy_reward"] == 0.0
    assert result["format_reward"] == 1.0
    assert result["weighted_format_reward"] == 0.2
    assert result["score"] == 1.0


# ---------------------------------------------------------------------------
# Strict mode — basic conditions ①②
# ---------------------------------------------------------------------------

def test_compute_format_reward_strict_returns_dict():
    """Strict mode with realistic multi-turn output: should return a dict with
    format_reward, retrieval_reward, sum_format_reward.

    Uses the real multi-turn structure where tool output is system-injected
    between ``\\nuser\\n`` and ``\\nassistant\\n`` boundaries.
    """
    response = (
        '<think>search first</think>'
        '<call_tool name="google_search">latest ai news</call_tool>'
        '\nuser\n'
        '<tool_response>\n<tool_output>\n'
        '<snippet id="s1">Title: A\nContent: AI news today</snippet>\n'
        '</tool_output>\n</tool_response>'
        '\nassistant\n'
        '\n<think>got result</think>'
        '<answer><cite id="s1">AI is in the news.</cite></answer>'
    )

    reward = search_r1_like_qa_em.compute_format_reward(
        response,
        mcp_parser_name="dr_tulu_xml",
        format_penalty="strict",
    )

    assert isinstance(reward, dict)
    assert reward["format_reward"] == 0.0      # conditions ①②③ all met
    assert reward["retrieval_reward"] == 0.2    # has <call_tool>
    assert reward["sum_format_reward"] == 0.2   # 0.0 + 0.2


def test_compute_format_reward_strict_penalizes_non_final_answer_block():
    """Strict mode: answer not at end → format_reward = -1."""
    response = (
        '<call_tool name="google_search">latest ai news</call_tool>'
        '<answer><cite id="s1">AI is in the news.</cite></answer>'
        '<think>extra trailing block</think>'
    )

    reward = search_r1_like_qa_em.compute_format_reward(
        response,
        mcp_parser_name="dr_tulu_xml",
        format_penalty="strict",
    )

    assert isinstance(reward, dict)
    assert reward["format_reward"] == -1.0
    assert reward["retrieval_reward"] == 0.2    # still has <call_tool>
    assert reward["sum_format_reward"] == -0.8  # -1.0 + 0.2


def test_compute_format_reward_strict_no_tool_call():
    """Strict mode: no tool call → retrieval_reward = 0, but format can still pass."""
    response = "<answer>42</answer>"

    reward = search_r1_like_qa_em.compute_format_reward(
        response,
        mcp_parser_name="dr_tulu_xml",
        format_penalty="strict",
    )

    assert isinstance(reward, dict)
    assert reward["format_reward"] == 0.0       # ①② met, ③ no call_tool so no fabrication check
    assert reward["retrieval_reward"] == 0.0    # no <call_tool>
    assert reward["sum_format_reward"] == 0.0   # 0.0 + 0.0


def test_compute_score_strict_applies_negative_format_penalty():
    """Strict mode compute_score: returns dict-based format metrics."""
    response = (
        '<think>search first</think>'
        '<call_tool name="google_search">latest ai news</call_tool>'
        '<answer><cite id="s1">wrong answer</cite></answer>'
        '<think>extra trailing block</think>'
    )
    ground_truth = {"target": ["correct answer"]}

    result = search_r1_like_qa_em.compute_score(response, ground_truth, format_penalty="strict")

    assert result["accuracy_reward"] == 0.0
    assert result["format_reward"] == -1.0
    assert result["retrieval_reward"] == 0.2
    assert result["sum_format_reward"] == -0.8
    assert "weighted_format_reward" not in result
    assert result["score"] == -0.8  # format_score used as fallback when answer is wrong


# ---------------------------------------------------------------------------
# Strict mode — condition ③: hallucinated tool detection (white-list based)
# ---------------------------------------------------------------------------

def test_strict_allowed_tools_pass():
    """Both allowed tools (google_search, browse_webpage)
    must NOT trigger the fabrication check.
    """
    response = (
        '<think>search</think>'
        '<call_tool name="google_search">query1</call_tool>'
        '\nuser\n'
        '<tool_response>\n<tool_output>\n<snippet id=s1>R1</snippet>\n</tool_output>\n</tool_response>'
        '\nassistant\n'
        '<think>browse</think>'
        '<call_tool name="browse_webpage">https://example.com</call_tool>'
        '\nuser\n'
        '<tool_response>\n<tool_output>\n<webpage id=w1>Page content</webpage>\n</tool_output>\n</tool_response>'
        '\nassistant\n'
        '<answer><cite id="s1">R1</cite> and <cite id="w1">Page content</cite>.</answer>'
    )

    reward = search_r1_like_qa_em.compute_format_reward(
        response,
        mcp_parser_name="dr_tulu_xml",
        format_penalty="strict",
    )

    assert isinstance(reward, dict)
    assert reward["format_reward"] == 0.0       # all tools are allowed
    assert reward["retrieval_reward"] == 0.2    # has valid tool calls
    assert reward["sum_format_reward"] == 0.2


def test_strict_hallucinated_tool_name_detected():
    """Model calls a tool not in the allowed list → fabrication detected."""
    response = (
        '<think>let me download it</think>'
        '<call_tool name="download_file">test.txt</call_tool>'
        '\nuser\n'
        '<tool_response>Error: tool not found</tool_response>'
        '\nassistant\n'
        '<answer>42</answer>'
    )

    reward = search_r1_like_qa_em.compute_format_reward(
        response,
        mcp_parser_name="dr_tulu_xml",
        format_penalty="strict",
    )

    assert isinstance(reward, dict)
    assert reward["format_reward"] == -1.0      # "download_file" not in allowed list
    assert reward["retrieval_reward"] == 0.2    # still has <call_tool> (extracted by regex)


def test_strict_mixed_valid_and_hallucinated_tools():
    """If ANY tool call uses a name outside the allowed list, it's a fabrication,
    even if other calls use valid names.
    """
    response = (
        '<call_tool name="google_search">query</call_tool>'
        '\nuser\n'
        '<tool_response>\n<tool_output>\n<snippet id=s1>Result</snippet>\n</tool_output>\n</tool_response>'
        '\nassistant\n'
        '<call_tool name="python">print(42)</call_tool>'
        '\nuser\n'
        '<tool_response>Error: unknown tool</tool_response>'
        '\nassistant\n'
        '<answer>42</answer>'
    )

    reward = search_r1_like_qa_em.compute_format_reward(
        response,
        mcp_parser_name="dr_tulu_xml",
        format_penalty="strict",
    )

    assert reward["format_reward"] == -1.0      # "python" not in allowed list


def test_strict_no_tool_call_skips_fabrication_check():
    """When there are no tool calls at all, the fabrication check is skipped entirely."""
    response = "<answer>The answer is 42</answer>"

    reward = search_r1_like_qa_em.compute_format_reward(
        response,
        mcp_parser_name="dr_tulu_xml",
        format_penalty="strict",
    )

    assert reward["format_reward"] == 0.0       # no queries → fabrication check skipped
    assert reward["retrieval_reward"] == 0.0


# ---------------------------------------------------------------------------
# _extract_model_generated_text edge cases
# ---------------------------------------------------------------------------

def test_extract_model_text_assistant_without_trailing_newline():
    """Edge case: the \\nassistant marker may lack a trailing \\n when
    the tokenizer merges it with the next token (observed in real rollouts
    where ``\\nassistant</think>`` appears instead of ``\\nassistant\\n``).

    _extract_model_generated_text() must still recover the model text
    from such turns so that fabrication detection and other checks work.
    """
    response = (
        '<think>t1</think>'
        '<call_tool name="google_search">q1</call_tool>'
        '\nuser\n'
        '<tool_response>search results</tool_response>'
        '\nassistant</think>\n\n'  # no \n between "assistant" and "</think>"
        '<answer>42</answer>'
    )

    model_text = search_r1_like_qa_em._extract_model_generated_text(response)

    # The model text from the last turn must be recovered (not dropped)
    assert '<answer>42</answer>' in model_text
    # System-injected content must be stripped
    assert '<tool_response>' not in model_text


def test_extract_model_text_hallucinated_tool_after_missing_newline():
    """If the model uses a hallucinated tool in a turn whose \\nassistant
    marker lacks the trailing \\n, detection must still catch it.
    """
    response = (
        '<think>t1</think>'
        '<call_tool name="google_search">q1</call_tool>'
        '\nuser\n'
        '<tool_response>real results</tool_response>'
        '\nassistant</think>\n\n'  # edge case boundary
        '<call_tool name="execute_code">import os</call_tool>'
        '<answer>42</answer>'
    )

    reward = search_r1_like_qa_em.compute_format_reward(
        response,
        mcp_parser_name="dr_tulu_xml",
        format_penalty="strict",
    )

    assert isinstance(reward, dict)
    assert reward["format_reward"] == -1.0   # "execute_code" not in allowed list
    assert reward["retrieval_reward"] == 0.2
