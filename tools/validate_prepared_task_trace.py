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
"""Validate PreparedTaskPipeline allocation, copy, and overlap evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger

_PREPARED_MARKERS = (
    "xllm.PreparedTask.Prepare",
    "xllm.PreparedTask.Launch",
    "xllm.PreparedTask.Consume",
)
_CONSISTENCY_CHECKS = (
    "marker_counts_equal",
    "metrics_available",
    "prepare_markers_match_metric_delta",
    "launch_markers_match_metric_delta",
    "consume_markers_match_metric_delta",
    "h2d_count_matches_metric_delta",
    "h2d_bytes_match_metric_delta",
)


def _read_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)
    if not isinstance(report, dict):
        raise ValueError("trace report root must be a JSON object")
    if report.get("schema_version") != 1:
        raise ValueError(f"unsupported trace report schema: {report.get('schema_version')}")
    return report


def _nested_dict(parent: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return value


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def validate_prepared_task_trace(
    report: dict[str, Any],
    arena_capacity_bytes: int,
    require_zero_cann_allocation: bool,
    allow_prepared_d2d: bool,
    require_prepare_device_overlap: bool,
    require_graph_replay: bool,
    require_graph_address_stability: bool,
) -> dict[str, Any]:
    """Return explicit acceptance checks for one analyzed trace report."""
    if arena_capacity_bytes <= 0:
        raise ValueError("arena capacity must be positive")

    checks: dict[str, bool] = {}
    failures: list[str] = []

    markers = _nested_dict(report, "markers", "report")
    marker_counts: dict[str, int] = {}
    for marker_name in _PREPARED_MARKERS:
        marker = _nested_dict(markers, marker_name, "markers")
        marker_counts[marker_name] = _integer(marker.get("count"), f"markers.{marker_name}.count")
    checks["prepared_markers_present"] = all(count > 0 for count in marker_counts.values())
    checks["prepared_marker_counts_equal"] = len(set(marker_counts.values())) == 1

    consistency = _nested_dict(report, "consistency_checks", "report")
    for check_name in _CONSISTENCY_CHECKS:
        checks[f"analyzer_{check_name}"] = consistency.get(check_name) is True
    if not allow_prepared_d2d:
        checks["prepared_staging_d2d_is_zero"] = consistency.get("prepared_d2d_metric_is_zero") is True

    copies = _nested_dict(report, "copies", "report")
    d2d_max_bytes = 0
    for operation_name, operation in copies.items():
        if "device to device" not in operation_name.lower():
            continue
        if not isinstance(operation, dict):
            raise ValueError(f"copies.{operation_name} must be an object")
        d2d_max_bytes = max(
            d2d_max_bytes,
            _integer(
                operation.get("workload_max_bytes", 0),
                f"copies.{operation_name}.workload_max_bytes",
            ),
        )
    checks["no_full_arena_d2d"] = d2d_max_bytes < arena_capacity_bytes

    if require_zero_cann_allocation:
        allocations = _nested_dict(report, "allocations", "report")
        allocation_count = 0
        for api_name, allocation in allocations.items():
            if not isinstance(allocation, dict):
                raise ValueError(f"allocations.{api_name} must be an object")
            allocation_count += _integer(
                allocation.get("workload_count", 0),
                f"allocations.{api_name}.workload_count",
            )
        checks["warmed_cann_allocation_is_zero"] = allocation_count == 0

    overlap = _nested_dict(
        report,
        "prepare_temporal_prior_launch_device_overlap",
        "report",
    )
    if require_prepare_device_overlap:
        checks["prepare_overlaps_preceding_task"] = (
            _integer(
                overlap.get("matched_prepare_count"),
                "prepare_temporal_prior_launch_device_overlap.matched_prepare_count",
            )
            > 0
            and _integer(
                overlap.get("overlapping_prepare_count"),
                "prepare_temporal_prior_launch_device_overlap.overlapping_prepare_count",
            )
            > 0
        )

    metrics = report.get("metrics")
    if require_graph_replay or require_graph_address_stability:
        if not isinstance(metrics, dict):
            raise ValueError("report.metrics must be available for Graph checks")
        delta = _nested_dict(metrics, "delta", "metrics")
        if require_graph_replay:
            checks["graph_replay_observed"] = (
                _integer(
                    delta.get("prepared_task_execution_total_graph_replay"),
                    "metrics.delta.prepared_task_execution_total_graph_replay",
                )
                > 0
            )
        if require_graph_address_stability:
            checks["graph_address_mismatches_are_zero"] = (
                _integer(
                    delta.get("prepared_task_graph_address_mismatches_total"),
                    "metrics.delta.prepared_task_graph_address_mismatches_total",
                )
                == 0
            )

    for check_name, passed in checks.items():
        if not passed:
            failures.append(check_name)
    return {
        "passed": not failures,
        "arena_capacity_bytes": arena_capacity_bytes,
        "marker_counts": marker_counts,
        "maximum_device_to_device_copy_bytes": d2d_max_bytes,
        "checks": checks,
        "failures": failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arena-capacity-bytes", required=True, type=int)
    parser.add_argument("--require-zero-cann-allocation", action="store_true")
    parser.add_argument("--allow-prepared-d2d", action="store_true")
    parser.add_argument("--require-prepare-device-overlap", action="store_true")
    parser.add_argument("--require-graph-replay", action="store_true")
    parser.add_argument("--require-graph-address-stability", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        report = _read_report(args.report)
        result = validate_prepared_task_trace(
            report,
            arena_capacity_bytes=args.arena_capacity_bytes,
            require_zero_cann_allocation=args.require_zero_cann_allocation,
            allow_prepared_d2d=args.allow_prepared_d2d,
            require_prepare_device_overlap=(args.require_prepare_device_overlap),
            require_graph_replay=args.require_graph_replay,
            require_graph_address_stability=(args.require_graph_address_stability),
        )
    except (OSError, ValueError, json.JSONDecodeError):
        logger.exception(f"failed to validate Prepared trace: {args.report}")
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        logger.error(f"Prepared trace acceptance failed: {result['failures']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
