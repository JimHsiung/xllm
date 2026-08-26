# Copyright 2026 The xLLM Authors. All Rights Reserved.
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
# ==============================================================================
"""Compare paired xLLM E2E throughput and client-latency results."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger


def _load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as result_file:
        result = json.load(result_file)
    if not isinstance(result, dict):
        raise ValueError(f"E2E result must be an object: {path}")
    return result


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    return float(value)


def _result_metrics(path: Path) -> dict[str, float]:
    result = _load_result(path)
    summary = result.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"missing summary object: {path}")
    total_requests = int(_number(summary.get("total_requests"), "total_requests"))
    successful_requests = int(_number(summary.get("successful_requests"), "successful_requests"))
    failed_requests = int(_number(summary.get("failed_requests"), "failed_requests"))
    if successful_requests != total_requests or failed_requests != 0:
        raise ValueError(f"E2E result contains failed requests: {path}")

    duration_seconds = _number(summary.get("duration_seconds"), "duration_seconds")
    total_tokens = _number(summary.get("total_tokens"), "total_tokens")
    completion_tokens = _number(summary.get("completion_tokens"), "completion_tokens")
    if duration_seconds <= 0 or total_tokens <= 0 or completion_tokens <= 0:
        raise ValueError(f"E2E duration and token counts must be positive: {path}")
    latency = summary.get("latency_seconds")
    if not isinstance(latency, dict):
        raise ValueError(f"missing latency summary: {path}")
    p50_seconds = _number(latency.get("p50"), "latency_seconds.p50")
    p95_seconds = _number(latency.get("p95"), "latency_seconds.p95")
    if p50_seconds <= 0 or p95_seconds <= 0:
        raise ValueError(f"latency percentiles must be positive: {path}")
    return {
        "duration_seconds": duration_seconds,
        "total_tokens": total_tokens,
        "completion_tokens": completion_tokens,
        "total_token_throughput": total_tokens / duration_seconds,
        "completion_token_throughput": completion_tokens / duration_seconds,
        "p50_latency_seconds": p50_seconds,
        "p95_latency_seconds": p95_seconds,
    }


def _ratio(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline performance metric must be positive")
    return candidate / baseline


def compare_e2e_performance_pairs(
    pairs: list[tuple[str, Path, Path]],
    min_total_token_throughput_ratio: float | None = None,
    min_completion_token_throughput_ratio: float | None = None,
    max_p50_latency_ratio: float | None = None,
    max_p95_latency_ratio: float | None = None,
) -> dict[str, Any]:
    """Aggregate paired run ratios and apply optional median gates."""
    if not pairs:
        raise ValueError("at least one performance pair is required")
    thresholds = {
        "min_total_token_throughput_ratio": min_total_token_throughput_ratio,
        "min_completion_token_throughput_ratio": (min_completion_token_throughput_ratio),
        "max_p50_latency_ratio": max_p50_latency_ratio,
        "max_p95_latency_ratio": max_p95_latency_ratio,
    }
    for threshold_name, threshold in thresholds.items():
        if threshold is not None and threshold <= 0:
            raise ValueError(f"{threshold_name} must be positive")

    pair_summaries: list[dict[str, Any]] = []
    ratio_values: dict[str, list[float]] = {
        "total_token_throughput_ratio": [],
        "completion_token_throughput_ratio": [],
        "p50_latency_ratio": [],
        "p95_latency_ratio": [],
    }
    for label, baseline_path, candidate_path in pairs:
        baseline_result = _load_result(baseline_path)
        candidate_result = _load_result(candidate_path)
        configuration_matches = baseline_result.get("configuration") == candidate_result.get("configuration")
        model_matches = baseline_result.get("model") == candidate_result.get("model")
        if not configuration_matches or not model_matches:
            raise ValueError(f"performance pair {label} does not use the same model and request configuration")
        baseline = _result_metrics(baseline_path)
        candidate = _result_metrics(candidate_path)
        if (
            baseline["total_tokens"] != candidate["total_tokens"]
            or baseline["completion_tokens"] != candidate["completion_tokens"]
        ):
            raise ValueError(f"performance pair {label} does not contain equal token counts")
        ratios = {
            "total_token_throughput_ratio": _ratio(
                candidate["total_token_throughput"],
                baseline["total_token_throughput"],
            ),
            "completion_token_throughput_ratio": _ratio(
                candidate["completion_token_throughput"],
                baseline["completion_token_throughput"],
            ),
            "p50_latency_ratio": _ratio(
                candidate["p50_latency_seconds"],
                baseline["p50_latency_seconds"],
            ),
            "p95_latency_ratio": _ratio(
                candidate["p95_latency_seconds"],
                baseline["p95_latency_seconds"],
            ),
        }
        for ratio_name, value in ratios.items():
            ratio_values[ratio_name].append(value)
        pair_summaries.append(
            {
                "label": label,
                "baseline": str(baseline_path),
                "candidate": str(candidate_path),
                "baseline_metrics": baseline,
                "candidate_metrics": candidate,
                "ratios": {name: round(value, 6) for name, value in ratios.items()},
            }
        )

    raw_median_ratios = {name: statistics.median(values) for name, values in ratio_values.items()}
    median_ratios = {name: round(value, 6) for name, value in raw_median_ratios.items()}
    checks: dict[str, bool] = {}
    if min_total_token_throughput_ratio is not None:
        checks["total_token_throughput"] = (
            raw_median_ratios["total_token_throughput_ratio"] >= min_total_token_throughput_ratio
        )
    if min_completion_token_throughput_ratio is not None:
        checks["completion_token_throughput"] = (
            raw_median_ratios["completion_token_throughput_ratio"] >= min_completion_token_throughput_ratio
        )
    if max_p50_latency_ratio is not None:
        checks["p50_latency"] = raw_median_ratios["p50_latency_ratio"] <= max_p50_latency_ratio
    if max_p95_latency_ratio is not None:
        checks["p95_latency"] = raw_median_ratios["p95_latency_ratio"] <= max_p95_latency_ratio
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "pair_count": len(pair_summaries),
        "thresholds": thresholds,
        "median_ratios": median_ratios,
        "checks": checks,
        "failures": failures,
        "pairs": pair_summaries,
    }


def _parse_pair(pair_spec: str) -> tuple[str, Path, Path]:
    if "=" not in pair_spec or "," not in pair_spec:
        raise ValueError("pair must use LABEL=BASELINE_PATH,CANDIDATE_PATH syntax")
    label, paths = pair_spec.split("=", 1)
    baseline_path, candidate_path = paths.split(",", 1)
    if not label or not baseline_path or not candidate_path:
        raise ValueError("pair label and paths must be non-empty")
    return label, Path(baseline_path), Path(candidate_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-total-token-throughput-ratio", type=float)
    parser.add_argument("--min-completion-token-throughput-ratio", type=float)
    parser.add_argument("--max-p50-latency-ratio", type=float)
    parser.add_argument("--max-p95-latency-ratio", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = compare_e2e_performance_pairs(
            [_parse_pair(pair_spec) for pair_spec in args.pair],
            min_total_token_throughput_ratio=(args.min_total_token_throughput_ratio),
            min_completion_token_throughput_ratio=(args.min_completion_token_throughput_ratio),
            max_p50_latency_ratio=args.max_p50_latency_ratio,
            max_p95_latency_ratio=args.max_p95_latency_ratio,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        logger.exception("failed to compare E2E performance results")
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        logger.error(f"E2E performance acceptance failed: {result['failures']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
