# Copyright 2026 Individual Contributor
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

from verl.utils.reward_score.rubric_gpt_judge import (
    FORMAT_REWARD_WEIGHT,
    _compute_format_reward,
)


def test_compute_format_reward_matches_dr_tulu_weights():
    response = (
        '<think>search first</think>'
        '<call_tool name="google_search">next olympics</call_tool>'
        '<tool_output><snippet id="s1">Title: Example</snippet></tool_output>'
        '<answer><cite id="s1">The next Olympics are in Italy.</cite></answer>'
    )

    reward = _compute_format_reward(response, format_penalty="easy")

    assert reward == 1.0
    assert FORMAT_REWARD_WEIGHT * reward == 0.2


def test_compute_format_reward_is_partial_when_only_answer_exists():
    response = "<answer>Just the answer.</answer>"

    reward = _compute_format_reward(response, format_penalty="easy")

    assert reward == 0.5


def test_compute_format_reward_supports_strict_success_case():
    """Strict mode returns a dict with format_reward, retrieval_reward, sum_format_reward.

    Uses realistic multi-turn structure where tool output is system-injected.
    """
    response = (
        '<call_tool name="google_search">next olympics</call_tool>'
        '\nuser\n'
        '<tool_response>\n<tool_output>\n'
        '<snippet id="s1">Title: Example</snippet>\n'
        '</tool_output>\n</tool_response>'
        '\nassistant\n'
        '\n<think>got it</think>'
        '<answer><cite id="s1">The next Olympics are in Italy.</cite></answer>'
    )

    reward = _compute_format_reward(response, format_penalty="strict")

    assert isinstance(reward, dict)
    assert reward["format_reward"] == 0.0
    assert reward["retrieval_reward"] == 0.2
    assert reward["sum_format_reward"] == 0.2


def test_compute_format_reward_supports_strict_failure_case():
    """Strict mode: answer not at end → format_reward = -1."""
    response = (
        '<call_tool name="google_search">next olympics</call_tool>'
        '<answer><cite id="s1">The next Olympics are in Italy.</cite></answer>'
        '<think>extra trailing block</think>'
    )

    reward = _compute_format_reward(response, format_penalty="strict")

    assert isinstance(reward, dict)
    assert reward["format_reward"] == -1.0
    assert reward["retrieval_reward"] == 0.2
    assert reward["sum_format_reward"] == -0.8
