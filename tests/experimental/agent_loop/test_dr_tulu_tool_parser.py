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
import json

import pytest

from verl.experimental.agent_loop.tool_agent_loop import _apply_tool_call_stop_strings
from verl.experimental.agent_loop.tool_parser import DrTuluXMLToolParser
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)


class DummyTokenizer:
    def decode(self, token_ids):
        return token_ids


@pytest.mark.asyncio
async def test_dr_tulu_xml_tool_parser():
    parser = DrTuluXMLToolParser(DummyTokenizer())
    tools = [
        OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="google_search",
                description="search",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query": OpenAIFunctionPropertySchema(type="string"),
                        "gl": OpenAIFunctionPropertySchema(type="string"),
                    },
                    required=["query"],
                ),
            ),
        ),
        OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="browse_webpage",
                description="browse",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={"url": OpenAIFunctionPropertySchema(type="string")},
                    required=["url"],
                ),
            ),
        ),
    ]
    text, function_calls = await parser.extract_tool_calls(
        '<think>need search</think><call_tool name="google_search" gl="us">latest ai news</call_tool>',
        tools,
    )
    assert text == "<think>need search</think>"
    assert len(function_calls) == 1
    assert function_calls[0].name == "google_search"
    assert json.loads(function_calls[0].arguments) == {"gl": "us", "query": "latest ai news"}

    _, browse_calls = await parser.extract_tool_calls(
        '<call_tool name="browse_webpage">https://example.com/article</call_tool>',
        tools,
    )
    assert len(browse_calls) == 1
    assert json.loads(browse_calls[0].arguments) == {"url": "https://example.com/article"}


def test_dr_tulu_xml_tool_parser_stop_strings():
    parser = DrTuluXMLToolParser(DummyTokenizer())

    assert parser.get_stop_strings() == ["</call_tool>"]
    assert parser.should_include_stop_strings_in_output() is True


def test_apply_tool_call_stop_strings_for_vllm():
    parser = DrTuluXMLToolParser(DummyTokenizer())

    sampling_params = _apply_tool_call_stop_strings(
        {"temperature": 0.7, "stop": ["</answer>"]},
        parser,
        rollout_name="vllm",
    )

    assert sampling_params["stop"] == ["</answer>", "</call_tool>"]
    assert sampling_params["include_stop_str_in_output"] is True
