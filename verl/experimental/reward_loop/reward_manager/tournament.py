# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Tournament-ranking reward manager.

This reward manager groups rollouts by query and runs a multi-round
tournament to determine relative quality, replacing the per-rollout
absolute GPT judge scoring.

It registers under the name ``"tournament"`` and can be selected via::

    reward.reward_manager.name=tournament

The manager overrides the ``run_batch`` method so that the
``RewardLoopWorker`` processes the entire batch at once (rather than
item-by-item), enabling cross-rollout comparison within the same query.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections import defaultdict
from typing import Any

from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _to_plain_dict(value: Any) -> dict:
    if not value:
        return {}
    try:
        from omegaconf import OmegaConf
        return dict(OmegaConf.to_container(value, resolve=True) or {})
    except Exception:
        return dict(value)


@register("tournament")
class TournamentRewardManager(RewardManagerBase):
    """Reward manager that uses tournament ranking across rollouts of the same query.

    Instead of scoring each rollout independently, this manager:
    1. Groups all rollouts by their query / prompt index.
    2. For each query-group, runs a multi-round tournament via GPT-4o.
    3. Maps cumulative tournament scores back to per-rollout rewards.

    The heavy lifting is in :func:`tournament_gpt_judge.compute_score_batch`.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        # We use compute_score only as a per-item fallback; the main path goes
        # through compute_score_batch from tournament_gpt_judge.
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        # Load tournament-specific config
        gpt_judge_cfg = _to_plain_dict(config.reward.get("gpt_judge", {}))
        reward_kwargs = _to_plain_dict(config.reward.get("reward_kwargs", {}))
        self._gpt_judge_cfg = gpt_judge_cfg
        self._reward_kwargs = reward_kwargs

    # ------------------------------------------------------------------
    # run_single: fallback for single-item calls
    # ------------------------------------------------------------------

    async def run_single(self, data: DataProto) -> dict:
        """Fall back to per-item scoring when called with a single item."""
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch.get("data_source", "unknown")
        reward_model = data_item.non_tensor_batch.get("reward_model", None)
        ground_truth = reward_model["ground_truth"] if isinstance(reward_model, dict) else None
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                ),
            )

        reward_extra_info = {}
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        return {"reward_score": score, "reward_extra_info": reward_extra_info}

    # ------------------------------------------------------------------
    # run_batch: the main entry-point for tournament ranking
    # ------------------------------------------------------------------

    async def run_batch(self, data: DataProto) -> list[dict]:
        """Process the entire batch, grouping rollouts by query for tournament.

        This method is called by ``RewardLoopWorker.compute_score_batch``
        when the reward manager has a ``run_batch`` method.  It receives
        a chunk of the full batch and is responsible for returning a list
        of per-item result dicts.
        """
        n = len(data)
        print(f"[DIAG] TournamentRewardManager.run_batch CALLED, n={n}", flush=True)
        if n == 0:
            return []

        # ---- Step 1: Decode all responses and collect metadata ----
        items_meta: list[dict[str, Any]] = []
        for i in range(n):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            data_source = data_item.non_tensor_batch.get("data_source", "unknown")
            reward_model = data_item.non_tensor_batch.get("reward_model", None)
            ground_truth = reward_model["ground_truth"] if isinstance(reward_model, dict) else None
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
            if tool_extra_fields is not None:
                extra_info.update(tool_extra_fields.items())

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            items_meta.append({
                "index": i,
                "data_source": data_source,
                "ground_truth": ground_truth,
                "extra_info": extra_info,
                "solution_str": response_str,
            })

        # ---- Step 2: Group by query (using ground_truth identity) ----
        # Rollouts for the same query share the same ground_truth dict.
        # We use the query string extracted from ground_truth as the grouping key.
        from verl.utils.reward_score.tournament_gpt_judge import (
            _extract_query_and_rubrics,
            compute_score_batch as tournament_compute_score_batch,
        )

        query_groups: dict[str, list[int]] = defaultdict(list)
        for idx, meta in enumerate(items_meta):
            gt = meta["ground_truth"]
            query, _ = _extract_query_and_rubrics(gt, extra_info=meta["extra_info"])
            if idx < 8:  # log first few for debugging
                print(f"[DIAG] run_batch item {idx}: gt_type={type(gt).__name__}, "
                      f"gt_is_dict={isinstance(gt, dict)}, "
                      f"query_repr={repr(query[:80]) if query else 'EMPTY'}", flush=True)
            query_groups[query].append(idx)

        print(f"[DIAG] run_batch query_groups: {len(query_groups)} unique queries, "
              f"group_sizes={[len(v) for v in query_groups.values()]}", flush=True)

        # ---- Step 3: Run tournament for each query group ----
        import asyncio

        results: list[dict | None] = [None] * n

        async def _process_query_group(query: str, indices: list[int]):
            group_items = [
                {
                    "solution_str": items_meta[i]["solution_str"],
                    "ground_truth": items_meta[i]["ground_truth"],
                    "extra_info": items_meta[i]["extra_info"],
                    "data_source": items_meta[i]["data_source"],
                }
                for i in indices
            ]

            if len(group_items) <= 1:
                print(f"[DIAG] _process_query_group: SINGLE-ITEM fallback for query={query[:60]!r}, "
                      f"len={len(group_items)}", flush=True)
                # Single rollout – fall back to per-item scoring
                for i, item_idx in enumerate(indices):
                    meta = items_meta[item_idx]
                    if self.is_async_reward_score:
                        result = await self.compute_score(
                            data_source=meta["data_source"],
                            solution_str=meta["solution_str"],
                            ground_truth=meta["ground_truth"],
                            extra_info=meta["extra_info"],
                        )
                    else:
                        result = await self.loop.run_in_executor(
                            None,
                            lambda m=meta: self.compute_score(
                                data_source=m["data_source"],
                                solution_str=m["solution_str"],
                                ground_truth=m["ground_truth"],
                                extra_info=m["extra_info"],
                            ),
                        )
                    reward_extra_info = {}
                    if isinstance(result, dict):
                        score = result["score"]
                        for key, value in result.items():
                            reward_extra_info[key] = value
                    else:
                        score = result
                        reward_extra_info["acc"] = score
                    results[item_idx] = {"reward_score": score, "reward_extra_info": reward_extra_info}
                return

            # Multi-rollout: use tournament ranking
            group_results = await tournament_compute_score_batch(
                items=group_items,
                gpt_judge=self._gpt_judge_cfg,
                reward_kwargs=self._reward_kwargs,
            )

            for i, item_idx in enumerate(indices):
                result_dict = group_results[i]
                reward_extra_info = dict(result_dict)
                score = result_dict["score"]
                results[item_idx] = {"reward_score": score, "reward_extra_info": reward_extra_info}

        # Run all query groups concurrently (within each group, tournament
        # rounds are sequential; but different queries are independent)
        tasks = [_process_query_group(q, idxs) for q, idxs in query_groups.items()]
        await asyncio.gather(*tasks)

        # Ensure no None results remain
        for i in range(n):
            if results[i] is None:
                logger.error("Tournament produced no result for item %d, using fallback score 0.", i)
                results[i] = {"reward_score": 0.0, "reward_extra_info": {"error": "tournament_missing_result"}}

        return results
