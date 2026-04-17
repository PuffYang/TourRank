from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


@dataclass(frozen=True)
class OODBenchmarkSpec:
    display_name: str
    slug: str
    eval_task: str
    file_name: str


def _to_plain_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_plain_python(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_plain_python(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_plain_python(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain_python(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _to_plain_python(value.tolist())
        except Exception:
            return value
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_key_value_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, raw_val = line.split(":", 1)
            key = key.strip()
            raw_val = raw_val.strip()
            if not key or not raw_val:
                continue
            try:
                metrics[key] = float(raw_val)
            except ValueError:
                continue
    return metrics


class OODValidationRunner:
    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.config.trainer.get("ood_validation", None)
        self.enabled = bool(self.config and self.config.get("enable", False))
        self._dataset_cache: dict[Path, Any] = {}
        self._agent_root = self._resolve_agent_root()

    def _resolve_agent_root(self) -> Path:
        return Path(__file__).resolve().parent / "ood_eval_vendor" / "agent"

    def run(self) -> dict[str, float]:
        if not self.enabled:
            return {}

        if self.trainer.global_steps == 0 and not self.config.get("run_on_initial_validation", False):
            return {}

        data_dir = Path(os.path.expanduser(self.config.data_dir)).resolve()
        if not data_dir.exists():
            return self._handle_error(FileNotFoundError(f"OOD validation data dir not found: {data_dir}"))

        metrics: dict[str, float] = {}
        summaries: dict[str, Any] = {}
        step_dir = self._resolve_step_dir()
        step_dir.mkdir(parents=True, exist_ok=True)

        for spec in self._iter_specs():
            dataset_path = data_dir / spec.file_name
            if not dataset_path.exists():
                self._maybe_raise(RuntimeError(f"OOD validation file not found: {dataset_path}"))
                continue

            try:
                generated_records = self._generate_records(dataset_path, spec)
                benchmark_dir = step_dir / spec.slug
                benchmark_dir.mkdir(parents=True, exist_ok=True)

                input_path = benchmark_dir / "generated.jsonl"
                self._write_jsonl(input_path, generated_records)

                summary = self._evaluate_generated_outputs(spec, input_path, benchmark_dir)
                summaries[spec.display_name] = summary

                primary_score = self._primary_score(spec, summary)
                if primary_score is not None:
                    metrics[f"val-ood/{spec.display_name}/score"] = float(primary_score)

                for key, value in self._auxiliary_scores(spec, summary).items():
                    metrics[f"val-ood/{spec.display_name}/{key}"] = float(value)

                metrics[f"val-ood/{spec.display_name}/num_examples"] = float(len(generated_records))
            except Exception as exc:
                self._maybe_raise(exc)

        if summaries:
            summary_path = step_dir / "summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "global_step": self.trainer.global_steps,
                        "metrics": metrics,
                        "benchmarks": summaries,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        return metrics

    def _resolve_step_dir(self) -> Path:
        default_local_dir = Path(os.path.expanduser(self.trainer.config.trainer.default_local_dir))
        if not default_local_dir.is_absolute():
            default_local_dir = (Path.cwd() / default_local_dir).resolve()
        return default_local_dir / f"global_step_{self.trainer.global_steps}" / self.config.output_subdir

    def _iter_specs(self) -> list[OODBenchmarkSpec]:
        return [
            OODBenchmarkSpec(
                display_name="DeepResearch Bench",
                slug="deepresearch_bench",
                eval_task="deep_research_bench",
                file_name=self.config.deep_research_bench_file,
            ),
        ]

    def _get_dataset(self, dataset_path: Path):
        if dataset_path in self._dataset_cache:
            return self._dataset_cache[dataset_path]

        from verl.trainer.main_ppo import create_rl_dataset

        dataset = create_rl_dataset(
            str(dataset_path),
            self.trainer.config.data,
            self.trainer.tokenizer,
            self.trainer.processor,
            is_train=False,
            max_samples=-1,
        )
        self._dataset_cache[dataset_path] = dataset
        return dataset

    def _generate_records(self, dataset_path: Path, spec: OODBenchmarkSpec) -> list[dict[str, Any]]:
        from verl.utils.dataset.rl_dataset import collate_fn

        dataset = self._get_dataset(dataset_path)
        batch_size = int(self.config.get("batch_size", 16))
        num_workers = self.trainer.config.data["dataloader_num_workers"]

        dataloader = StatefulDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        records: list[dict[str, Any]] = []
        for test_data in dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_gen_batch = self.trainer._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.trainer.tokenizer.eos_token_id,
                "pad_token_id": self.trainer.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.trainer.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.trainer.global_steps,
            }

            size_divisor = self.trainer.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.trainer.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            merged_batch = test_batch.union(test_output_gen_batch)

            output_ids = merged_batch.batch["responses"]
            output_texts = [
                self.trainer._format_rollout_output_text(
                    self.trainer.tokenizer.decode(ids, skip_special_tokens=True)
                )
                for ids in output_ids
            ]

            extra_infos = merged_batch.non_tensor_batch.get("extra_info")
            full_traces = merged_batch.non_tensor_batch.get("full_traces")
            sample_uids = merged_batch.non_tensor_batch.get("uid")

            for idx, output_text in enumerate(output_texts):
                extra_info = _to_plain_python(extra_infos[idx]) if extra_infos is not None else {}
                if not isinstance(extra_info, dict):
                    extra_info = {}

                original_data = extra_info.get("original_data", {})
                problem = extra_info.get("problem") or self._problem_from_original_data(original_data)
                example_id = extra_info.get("example_id") or extra_info.get("orig_id") or str(sample_uids[idx])

                records.append(
                    {
                        "example_id": str(example_id),
                        "problem": str(problem or ""),
                        "final_response": output_text,
                        "pred_answer": output_text,
                        "full_traces": self._normalize_full_traces(
                            full_traces[idx] if full_traces is not None else None
                        ),
                        "original_data": _to_plain_python(original_data),
                    }
                )

        return records

    def _normalize_full_traces(self, value: Any) -> dict[str, Any]:
        plain = _to_plain_python(value)
        if not isinstance(plain, dict):
            return {"generated_text": "", "tool_calls": []}

        generated_text = plain.get("generated_text", "")
        tool_calls = plain.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []

        return {
            **plain,
            "generated_text": generated_text if isinstance(generated_text, str) else "",
            "tool_calls": [_to_plain_python(item) for item in tool_calls],
        }

    def _problem_from_original_data(self, original_data: Any) -> str:
        data = _to_plain_python(original_data)
        if isinstance(data, dict):
            for key in ("problem", "query", "question", "prompt"):
                value = data.get(key)
                if isinstance(value, str):
                    return value
                if key == "prompt" and isinstance(value, list) and len(value) > 0:
                    last_message = value[-1]
                    if isinstance(last_message, dict):
                        content = last_message.get("content")
                        if isinstance(content, str):
                            return content
        return ""

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _evaluate_generated_outputs(
        self, spec: OODBenchmarkSpec, input_path: Path, benchmark_dir: Path
    ) -> dict[str, Any]:
        if spec.eval_task == "deep_research_bench":
            return self._evaluate_deep_research_bench(spec, input_path, benchmark_dir)
        raise ValueError(f"Unsupported OOD benchmark task: {spec.eval_task}")

    def _evaluate_deep_research_bench(
        self, spec: OODBenchmarkSpec, input_path: Path, benchmark_dir: Path
    ) -> dict[str, Any]:
        script_path = self._agent_root / "evaluation" / "deep_research_bench_eval" / "run_eval.py"
        command = [
            sys.executable,
            str(script_path),
            "--input_file",
            str(input_path),
            "--task_name",
            f"{spec.slug}_step_{self.trainer.global_steps}",
            "--output_dir",
            str(benchmark_dir),
            "--max_workers",
            str(self.config.deep_research_bench_max_workers),
        ]
        self._run_subprocess(command, cwd=self._agent_root, log_dir=benchmark_dir)

        summary_path = benchmark_dir / "summary.json"
        if summary_path.exists():
            return _read_json(summary_path)

        race_metrics = _read_key_value_metrics(
            benchmark_dir / "race" / f"{spec.slug}_step_{self.trainer.global_steps}" / "race_result.txt"
        )
        return {
            "score": race_metrics.get("overall_score"),
            "race": race_metrics,
        }

    def _run_subprocess(self, command: list[str], cwd: Path, log_dir: Path) -> None:
        env = os.environ.copy()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")

        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)

        if completed.returncode != 0:
            raise RuntimeError(
                f"OOD evaluation command failed with code {completed.returncode}: {' '.join(command)}"
            )

    def _primary_score(self, spec: OODBenchmarkSpec, summary: dict[str, Any]) -> Optional[float]:
        if spec.eval_task == "deep_research_bench":
            race_metrics = summary.get("race", {})
            if isinstance(race_metrics, dict) and isinstance(race_metrics.get("overall_score"), (int, float)):
                return float(race_metrics["overall_score"])
            return None
        return None

    def _auxiliary_scores(self, spec: OODBenchmarkSpec, summary: dict[str, Any]) -> dict[str, float]:
        if spec.eval_task != "deep_research_bench":
            return {}

        aux_metrics: dict[str, float] = {}
        race_metrics = summary.get("race", {})
        if isinstance(race_metrics, dict):
            for metric_name, metric_value in race_metrics.items():
                if isinstance(metric_value, (int, float)):
                    aux_metrics[f"race_{metric_name}"] = float(metric_value)

        primary_score = self._primary_score(spec, summary)
        if primary_score is not None:
            for duplicate_key in ("race_overall_score", "overall_score"):
                duplicate_val = aux_metrics.get(duplicate_key)
                if duplicate_val is not None and abs(duplicate_val - primary_score) < 1e-12:
                    aux_metrics.pop(duplicate_key, None)

        return aux_metrics

    def _maybe_raise(self, exc: Exception) -> None:
        if self.config.get("fail_on_error", False):
            raise exc
        print(f"[OODValidationRunner] {exc}", file=sys.stderr)

    def _handle_error(self, exc: Exception) -> dict[str, float]:
        self._maybe_raise(exc)
        return {}
