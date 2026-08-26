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
"""Aggregate independently validated PreparedTaskPipeline rank traces."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger


def _load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)
    if not isinstance(report, dict):
        raise ValueError(f"rank trace report must be an object: {path}")
    if not isinstance(report.get("passed"), bool):
        raise ValueError(f"rank trace report has no boolean passed field: {path}")
    marker_counts = report.get("marker_counts")
    if not isinstance(marker_counts, dict) or not marker_counts:
        raise ValueError(f"rank trace report has no marker counts: {path}")
    return report


def aggregate_prepared_task_trace_validations(
    rank_reports: list[tuple[int, Path]],
) -> dict[str, Any]:
    """Require one passing validation report for each contiguous rank."""
    if not rank_reports:
        raise ValueError("at least one rank trace report is required")
    reports_by_rank: dict[int, tuple[Path, dict[str, Any]]] = {}
    for rank, path in rank_reports:
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if rank in reports_by_rank:
            raise ValueError(f"duplicate rank trace report: {rank}")
        reports_by_rank[rank] = (path, _load_report(path))

    expected_ranks = list(range(len(reports_by_rank)))
    actual_ranks = sorted(reports_by_rank)
    if actual_ranks != expected_ranks:
        raise ValueError(f"rank trace reports must be contiguous from zero: {actual_ranks}")

    aggregate_marker_counts: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    maximum_d2d_bytes = 0
    for rank in actual_ranks:
        path, report = reports_by_rank[rank]
        marker_counts = report["marker_counts"]
        for marker_name, count in marker_counts.items():
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(f"rank {rank} marker count must be an integer: {marker_name}")
            aggregate_marker_counts[marker_name] = aggregate_marker_counts.get(marker_name, 0) + count
        d2d_bytes = report.get("maximum_device_to_device_copy_bytes", 0)
        if isinstance(d2d_bytes, bool) or not isinstance(d2d_bytes, int):
            raise ValueError(f"rank {rank} maximum D2D copy bytes must be an integer")
        maximum_d2d_bytes = max(maximum_d2d_bytes, d2d_bytes)
        rank_failures = report.get("failures", [])
        if not isinstance(rank_failures, list):
            raise ValueError(f"rank {rank} failures must be a list")
        if not report["passed"]:
            failures.append(
                {
                    "rank": rank,
                    "report": str(path),
                    "failures": rank_failures,
                }
            )
        summaries.append(
            {
                "rank": rank,
                "report": str(path),
                "passed": report["passed"],
                "marker_counts": marker_counts,
                "maximum_device_to_device_copy_bytes": d2d_bytes,
            }
        )
    return {
        "passed": not failures,
        "rank_count": len(summaries),
        "aggregate_marker_counts": aggregate_marker_counts,
        "maximum_device_to_device_copy_bytes": maximum_d2d_bytes,
        "failures": failures,
        "ranks": summaries,
    }


def _parse_rank_report(specification: str) -> tuple[int, Path]:
    if "=" not in specification:
        raise ValueError("rank report must use RANK=PATH syntax")
    rank_text, path_text = specification.split("=", 1)
    if not rank_text or not path_text:
        raise ValueError("rank and report path must be non-empty")
    return int(rank_text), Path(path_text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-report", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = aggregate_prepared_task_trace_validations([_parse_rank_report(spec) for spec in args.rank_report])
    except (OSError, ValueError, json.JSONDecodeError):
        logger.exception("failed to aggregate Prepared rank trace reports")
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        logger.error(f"Prepared rank trace acceptance failed: {result['failures']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
