# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("tournament")
class TournamentRewardManager(RewardManagerBase):
    """Tournament ranking reward manager for group RL (e.g., GRPO)."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)

        reward_kwargs = config.reward.get("reward_kwargs", {}) or {}

        self.n_rollouts = int(reward_kwargs.get("n_rollouts", config.actor_rollout_ref.rollout.n))
        self.num_groups = int(reward_kwargs.get("num_groups", 8))
        self.num_winners_per_group = int(reward_kwargs.get("num_winners_per_group", 2))
        self.target_finalists = int(reward_kwargs.get("target_finalists", 2))
        self.score_increment = float(reward_kwargs.get("score_increment", 1.0))
        self.reward_mapping_method = str(reward_kwargs.get("reward_mapping_method", "identity")).lower()

        self.max_concurrency = int(reward_kwargs.get("max_concurrency", 8))
        self.judge_model_name = str(reward_kwargs.get("judge_model_name", "gpt-4o"))
        self.judge_temperature = float(reward_kwargs.get("judge_temperature", 0.2))
        self.judge_max_tokens = int(reward_kwargs.get("judge_max_tokens", 1024))
        self.judge_timeout = float(reward_kwargs.get("judge_timeout", 200))
        self.judge_max_retries = int(reward_kwargs.get("judge_max_retries", 3))
        self.judge_retry_sleep = float(reward_kwargs.get("judge_retry_sleep", 1.5))

        self.azure_api_key = str(reward_kwargs.get("judge_api_key", "b1b6cfd6240c446dbbe8ca087ca7fc02"))
        self.azure_api_version = str(reward_kwargs.get("judge_api_version", "2024-06-01"))
        self.azure_endpoint = str(
            reward_kwargs.get(
                "judge_azure_endpoint",
                "https://runway.devops.xiaohongshu.com/openai/chat/completions?api-version=2024-06-01",
            )
        )

        if self.num_groups <= 0:
            raise ValueError(f"num_groups must be > 0, got {self.num_groups}")
        if self.num_winners_per_group <= 0:
            raise ValueError(f"num_winners_per_group must be > 0, got {self.num_winners_per_group}")
        if self.target_finalists <= 0:
            raise ValueError(f"target_finalists must be > 0, got {self.target_finalists}")
        if self.max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be > 0, got {self.max_concurrency}")

        self._thread_local = threading.local()

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        outputs = await self.run_batch(data)
        return outputs[0]

    async def run_batch(self, data: DataProto) -> list[dict[str, Any]]:
        return await self.loop.run_in_executor(None, lambda: self._run_batch_sync(data))

    def _run_batch_sync(self, data: DataProto) -> list[dict[str, Any]]:
        if len(data) == 0:
            return []

        batch_items = self._collect_batch_items(data)
        groups = self._group_batch_items(batch_items)

        scores = [0.0 for _ in range(len(batch_items))]
        rewards = [0.0 for _ in range(len(batch_items))]
        rounds = [0 for _ in range(len(batch_items))]
        fallbacks = [False for _ in range(len(batch_items))]
        errors = ["" for _ in range(len(batch_items))]

        for _, indices in groups.items():
            group_items = [batch_items[i] for i in indices]
            group_scores, group_rewards, group_rounds, group_fallback, group_error = self._score_group(group_items)

            for local_idx, global_idx in enumerate(indices):
                scores[global_idx] = float(group_scores[local_idx])
                rewards[global_idx] = float(group_rewards[local_idx])
                rounds[global_idx] = int(group_rounds)
                fallbacks[global_idx] = bool(group_fallback)
                errors[global_idx] = str(group_error)

        outputs = []
        for i, item in enumerate(batch_items):
            reward_extra_info = {
                "acc": rewards[i],
                "tournament_raw_score": scores[i],
                "tournament_reward": rewards[i],
                "tournament_rounds": rounds[i],
                "tournament_group_size": item["group_size"],
                "tournament_fallback": fallbacks[i],
                "tournament_error": errors[i],
            }
            outputs.append({"reward_score": rewards[i], "reward_extra_info": reward_extra_info})

        return outputs

    def _collect_batch_items(self, data: DataProto) -> list[dict[str, Any]]:
        batch_items: list[dict[str, Any]] = []

        responses = data.batch["responses"]
        response_length = responses.shape[-1]
        attention_mask = data.batch["attention_mask"]

        prompt_ids = data.batch.get("prompts", None)
        prompt_length = prompt_ids.shape[-1] if prompt_ids is not None else 0

        for i in range(len(data)):
            valid_response_length = int(attention_mask[i][-response_length:].sum().item())
            response_ids = responses[i][:valid_response_length]
            response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            prompt_str = ""
            if prompt_ids is not None:
                valid_prompt_length = int(attention_mask[i][:prompt_length].sum().item())
                valid_prompt_ids = prompt_ids[i][-valid_prompt_length:]
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            non_tensor = {k: data.non_tensor_batch[k][i] for k in data.non_tensor_batch.keys()}

            query = self._extract_query(non_tensor, prompt_str)
            rubric = self._extract_rubric(non_tensor, query)
            uid = self._extract_uid(non_tensor, query, i)

            batch_items.append(
                {
                    "batch_idx": i,
                    "uid": uid,
                    "query": query,
                    "rubric": rubric,
                    "response": response_str,
                    "group_size": 1,
                }
            )

        return batch_items

    def _group_batch_items(self, batch_items: list[dict[str, Any]]) -> OrderedDict[str, list[int]]:
        groups: OrderedDict[str, list[int]] = OrderedDict()
        for i, item in enumerate(batch_items):
            groups.setdefault(item["uid"], []).append(i)

        for _, indices in groups.items():
            size = len(indices)
            for idx in indices:
                batch_items[idx]["group_size"] = size

        return groups

    def _score_group(self, group_items: list[dict[str, Any]]) -> tuple[list[float], list[float], int, bool, str]:
        n = len(group_items)
        if n == 1:
            return [0.0], self._map_scores_to_rewards([0.0]), 0, False, ""

        if self.n_rollouts > 0 and n != self.n_rollouts:
            logger.warning(
                "Tournament group size mismatch: expected n_rollouts=%s, got %s. Continue with actual group size.",
                self.n_rollouts,
                n,
            )

        scores = [0.0 for _ in range(n)]
        contestants = list(range(n))
        rounds = 0
        has_fallback = False
        last_error = ""

        query = str(group_items[0]["query"])
        rubric = group_items[0]["rubric"]
        responses = [str(item["response"]) for item in group_items]

        max_rounds = max(1, n * 2)
        while len(contestants) > self.target_finalists and rounds < max_rounds:
            round_groups = self._split_groups(contestants, self.num_groups)
            group_results = self._judge_round(
                query=query,
                rubric=rubric,
                responses=responses,
                round_groups=round_groups,
            )

            next_contestants: list[int] = []
            for result in group_results:
                winners = result["winners"]
                for winner in winners:
                    scores[winner] += self.score_increment
                next_contestants.extend(winners)

                if result["fallback"]:
                    has_fallback = True
                if result["error"]:
                    last_error = result["error"]

            # deduplicate while preserving order
            next_contestants = list(dict.fromkeys(next_contestants))

            # guard against no reduction
            if len(next_contestants) >= len(contestants):
                keep = max(self.target_finalists, len(contestants) // 2)
                ranked = sorted(contestants, key=lambda x: (-scores[x], x))
                next_contestants = ranked[:keep]

            contestants = next_contestants
            rounds += 1

        rewards = self._map_scores_to_rewards(scores)
        return scores, rewards, rounds, has_fallback, last_error

    def _split_groups(self, contestants: list[int], num_groups: int) -> list[list[int]]:
        if len(contestants) == 0:
            return []

        group_count = min(max(1, num_groups), len(contestants))
        base_size = len(contestants) // group_count
        remainder = len(contestants) % group_count

        groups: list[list[int]] = []
        cursor = 0
        for i in range(group_count):
            cur_size = base_size + (1 if i < remainder else 0)
            if cur_size <= 0:
                continue
            groups.append(contestants[cursor : cursor + cur_size])
            cursor += cur_size

        return groups

    def _judge_round(
        self,
        query: str,
        rubric: dict[str, Any],
        responses: list[str],
        round_groups: list[list[int]],
    ) -> list[dict[str, Any]]:
        if not round_groups:
            return []

        worker_num = min(self.max_concurrency, len(round_groups))
        results: list[dict[str, Any]] = [None for _ in range(len(round_groups))]  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=worker_num) as executor:
            futures = {}
            for group_id, contestants in enumerate(round_groups):
                futures[
                    executor.submit(
                        self._judge_group_with_retry,
                        group_id,
                        query,
                        rubric,
                        responses,
                        contestants,
                    )
                ] = group_id

            for future in as_completed(futures):
                group_id = futures[future]
                try:
                    results[group_id] = future.result()
                except Exception as exc:  # extremely defensive
                    winners_needed = min(self.num_winners_per_group, len(round_groups[group_id]))
                    fallback_winners = round_groups[group_id][:winners_needed]
                    results[group_id] = {
                        "group_id": group_id,
                        "winners": fallback_winners,
                        "fallback": True,
                        "error": str(exc),
                    }

        return results

    def _judge_group_with_retry(
        self,
        group_id: int,
        query: str,
        rubric: dict[str, Any],
        responses: list[str],
        contestants: list[int],
    ) -> dict[str, Any]:
        winners_needed = min(self.num_winners_per_group, len(contestants))

        if len(contestants) <= winners_needed:
            return {
                "group_id": group_id,
                "winners": contestants,
                "fallback": False,
                "error": "",
            }

        last_error = ""
        for attempt in range(1, self.judge_max_retries + 1):
            try:
                judge_result = self._judge_group_once(
                    group_id=group_id,
                    query=query,
                    rubric=rubric,
                    responses=responses,
                    contestants=contestants,
                    winners_needed=winners_needed,
                )
                return {
                    "group_id": group_id,
                    "winners": judge_result,
                    "fallback": False,
                    "error": "",
                }
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.judge_max_retries:
                    time.sleep(self.judge_retry_sleep * attempt)

        # fallback: keep deterministic top-k by original order in this group
        fallback_winners = contestants[:winners_needed]
        return {
            "group_id": group_id,
            "winners": fallback_winners,
            "fallback": True,
            "error": last_error,
        }

    def _judge_group_once(
        self,
        group_id: int,
        query: str,
        rubric: dict[str, Any],
        responses: list[str],
        contestants: list[int],
        winners_needed: int,
    ) -> list[int]:
        client = self._get_client()

        rubric_items = rubric.get("rubrics", []) if isinstance(rubric, dict) else []
        rubric_text_lines = []
        for idx, item in enumerate(rubric_items):
            title = str(item.get("title", ""))
            description = str(item.get("description", ""))
            weight = item.get("weight", 1)
            rubric_text_lines.append(
                f"{idx + 1}. title: {title}\\n   description: {description}\\n   weight: {weight}"
            )
        rubric_text = "\\n".join(rubric_text_lines) if rubric_text_lines else "No rubric items provided."

        candidates = []
        for local_idx, global_idx in enumerate(contestants):
            candidates.append(
                f"Candidate {local_idx} (global_rollout_index={global_idx}):\\n{responses[global_idx]}"
            )
        candidates_text = "\\n\\n".join(candidates)

        prompt = f"""You are an expert judge for deep-research responses.

You must compare candidate rollouts for the SAME user query.
Use ALL rubric items below. Each item has title, description, and weight.
Weight is a relative importance signal (larger weight => more important), NOT a normalized coefficient.

Query:
{query}

Rubric:
{rubric_text}

Candidates in this group:
{candidates_text}

Task:
1) For each candidate, assign a final weighted total score in [0, 100], using the rubric and emphasizing high-weight criteria.
2) Select exactly {winners_needed} winners (best candidates) by relative comparison within this group.
3) Winners must be listed by candidate_id (the local Candidate index shown above).

Return JSON only with this schema:
{{
  "group_id": {group_id},
  "winners": [<candidate_id>, ...],
  "candidate_scores": [
    {{"candidate_id": 0, "score": 0.0}}
  ],
  "reason": "short reason"
}}
"""

        resp = client.chat.completions.create(
            model=self.judge_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict JSON generator. Return only valid JSON, no markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.judge_temperature,
            max_tokens=self.judge_max_tokens,
            timeout=self.judge_timeout,
        )

        content = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(self._strip_markdown_json_fence(content))

        winners = parsed.get("winners", [])
        if not isinstance(winners, list):
            raise ValueError("winners must be a list")

        valid_local_ids = set(range(len(contestants)))
        parsed_winners: list[int] = []
        for item in winners:
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, np.integer)):
                local_id = int(item)
            elif isinstance(item, str) and item.strip().isdigit():
                local_id = int(item.strip())
            else:
                continue
            if local_id in valid_local_ids and local_id not in parsed_winners:
                parsed_winners.append(local_id)

        # fill winners using candidate_scores if needed
        if len(parsed_winners) < winners_needed:
            score_items = parsed.get("candidate_scores", [])
            scored = []
            if isinstance(score_items, list):
                for score_item in score_items:
                    if not isinstance(score_item, dict):
                        continue
                    c_id = score_item.get("candidate_id")
                    score = score_item.get("score")
                    if isinstance(c_id, (int, np.integer)) and int(c_id) in valid_local_ids:
                        try:
                            score_val = float(score)
                        except Exception:
                            score_val = 0.0
                        scored.append((int(c_id), score_val))
            scored.sort(key=lambda x: (-x[1], x[0]))
            for c_id, _ in scored:
                if c_id not in parsed_winners:
                    parsed_winners.append(c_id)
                if len(parsed_winners) >= winners_needed:
                    break

        # still not enough, fallback by local id order
        if len(parsed_winners) < winners_needed:
            for local_id in range(len(contestants)):
                if local_id not in parsed_winners:
                    parsed_winners.append(local_id)
                if len(parsed_winners) >= winners_needed:
                    break

        parsed_winners = parsed_winners[:winners_needed]
        return [contestants[local_id] for local_id in parsed_winners]

    def _map_scores_to_rewards(self, scores: list[float]) -> list[float]:
        if len(scores) == 0:
            return []

        method = self.reward_mapping_method
        values = np.array(scores, dtype=np.float32)

        if method == "identity":
            mapped = values
        elif method == "minmax":
            min_v = float(values.min())
            max_v = float(values.max())
            if abs(max_v - min_v) < 1e-8:
                mapped = np.zeros_like(values)
            else:
                mapped = (values - min_v) / (max_v - min_v)
        elif method == "zscore":
            mean_v = float(values.mean())
            std_v = float(values.std())
            mapped = (values - mean_v) / (std_v + 1e-6)
        elif method == "rank":
            order = np.argsort(-values)
            ranks = np.empty_like(order, dtype=np.float32)
            for pos, idx in enumerate(order):
                ranks[idx] = float(pos)
            denom = max(1.0, float(len(values) - 1))
            mapped = 1.0 - (ranks / denom)
        else:
            raise ValueError(
                f"Unsupported reward_mapping_method={self.reward_mapping_method}. "
                "Supported: identity|minmax|zscore|rank"
            )

        return mapped.astype(np.float32).tolist()

    def _extract_uid(self, non_tensor: dict[str, Any], query: str, fallback_idx: int) -> str:
        uid = non_tensor.get("uid", None)
        if uid is None:
            if isinstance(query, str) and query.strip():
                return f"query::{query.strip()}"
            return f"query_group_{fallback_idx}"
        return str(uid)

    def _extract_query(self, non_tensor: dict[str, Any], prompt_str: str) -> str:
        query = non_tensor.get("query", None)
        if isinstance(query, str) and query.strip():
            return query.strip()

        reward_model = non_tensor.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth", None)
            if isinstance(ground_truth, dict):
                gt_query = ground_truth.get("query", None)
                if isinstance(gt_query, str) and gt_query.strip():
                    return gt_query.strip()

        raw_prompt = non_tensor.get("raw_prompt", None)
        if isinstance(raw_prompt, (list, tuple)):
            for msg in raw_prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

        if prompt_str.strip():
            return prompt_str.strip()

        return ""

    def _extract_rubric(self, non_tensor: dict[str, Any], query: str) -> dict[str, Any]:
        for key in ["rubric", "rubrics"]:
            value = non_tensor.get(key, None)
            rubric = self._normalize_rubric(value, query)
            if rubric is not None:
                return rubric

        reward_model = non_tensor.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth", None)
            rubric = self._normalize_rubric(ground_truth, query)
            if rubric is not None:
                return rubric

        extra_info = non_tensor.get("extra_info", {})
        if isinstance(extra_info, dict):
            for key in ["rubric", "rubrics"]:
                rubric = self._normalize_rubric(extra_info.get(key, None), query)
                if rubric is not None:
                    return rubric

        return {"query": query, "rubrics": []}

    def _normalize_rubric(self, value: Any, query: str) -> dict[str, Any] | None:
        if value is None:
            return None

        if isinstance(value, dict):
            if "rubrics" in value and isinstance(value.get("rubrics"), list):
                q = value.get("query", query)
                return {"query": str(q), "rubrics": value.get("rubrics", [])}
            if "title" in value and "description" in value and "weight" in value:
                return {"query": query, "rubrics": [value]}

        if isinstance(value, list):
            return {"query": query, "rubrics": value}

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
                return self._normalize_rubric(parsed, query)
            except Exception:
                return None

        return None

    def _strip_markdown_json_fence(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return cleaned.strip()

    def _get_client(self):
        client = getattr(self._thread_local, "client", None)
        if client is not None:
            return client

        try:
            from openai import AzureOpenAI
        except Exception as exc:  # pragma: no cover - import-time env issue
            raise ImportError("openai package is required for tournament reward manager") from exc

        client = AzureOpenAI(
            api_key=self.azure_api_key,
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint,
        )
        self._thread_local.client = client
        return client
