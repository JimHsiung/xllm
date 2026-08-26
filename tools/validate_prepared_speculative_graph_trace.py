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
"""Validate accepted-length and Slot coverage for speculative Prepared Graph."""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger

_EVENT_NAME = "prepared_speculative_graph_execution"
_EVENT_PATTERN = re.compile(
    rf"\bevent={_EVENT_NAME} "
    r"mode=(?P<mode>eager|capture|replay) "
    r"slot_id=(?P<slot_id>-?\d+) "
    r"graph_key=(?P<graph_key>\d+) "
    r"graph_warmup=(?P<graph_warmup>[01]) "
    r"accepted_length=(?P<accepted_length>\d+) "
    r"verify_width=(?P<verify_width>\d+) "
    r"static_graph_tasks_prepared=(?P<static_graph_tasks_prepared>[01])\s*$"
)


def _parse_event_line(line: str, path: Path, line_number: int) -> dict[str, Any]:
    match = _EVENT_PATTERN.search(line)
    if match is None:
        raise ValueError(f"malformed {_EVENT_NAME} record: {path}:{line_number}: {line.rstrip()}")
    return {
        "mode": match.group("mode"),
        "slot_id": int(match.group("slot_id")),
        "graph_key": int(match.group("graph_key")),
        "graph_warmup": match.group("graph_warmup") == "1",
        "accepted_length": int(match.group("accepted_length")),
        "verify_width": int(match.group("verify_width")),
        "static_graph_tasks_prepared": (match.group("static_graph_tasks_prepared") == "1"),
    }


def parse_prepared_speculative_graph_trace(path: Path) -> list[dict[str, Any]]:
    """Parse speculative Prepared Graph execution records in source order."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            if f"event={_EVENT_NAME}" not in line:
                continue
            records.append(_parse_event_line(line, path, line_number))
    if not records:
        raise ValueError(f"no {_EVENT_NAME} records found in {path}")
    return records


def _failure(label: str, check: str, detail: str) -> dict[str, str]:
    return {"label": label, "check": check, "detail": detail}


def validate_prepared_speculative_graph_trace(
    path: Path,
    label: str,
    verify_width: int,
    slot_count: int,
    require_runtime_replay: bool = False,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate one rank's warmup captures and runtime replay provenance."""
    if verify_width <= 0:
        raise ValueError("verify_width must be positive")
    if slot_count <= 0:
        raise ValueError("slot_count must be positive")

    records = parse_prepared_speculative_graph_trace(path)
    failures: list[dict[str, str]] = []
    capture_counts: Counter[tuple[int, int]] = Counter()
    runtime_replay_counts: Counter[tuple[int, int]] = Counter()
    capture_graph_keys: dict[tuple[int, int], set[int]] = {}
    runtime_replay_graph_keys: Counter[tuple[int, int, int]] = Counter()
    mode_counts: Counter[str] = Counter()

    for record in records:
        mode = str(record["mode"])
        slot_id = int(record["slot_id"])
        accepted_length = int(record["accepted_length"])
        record_verify_width = int(record["verify_width"])
        graph_key = int(record["graph_key"])
        graph_warmup = bool(record["graph_warmup"])
        mode_counts[mode] += 1

        if record_verify_width != verify_width:
            failures.append(
                _failure(
                    label,
                    "verify_width",
                    f"observed {record_verify_width}, expected {verify_width}",
                )
            )
        if slot_id < 0 or slot_id >= slot_count:
            failures.append(
                _failure(
                    label,
                    "slot_range",
                    f"observed slot {slot_id}, expected [0, {slot_count - 1}]",
                )
            )
        if accepted_length < 1 or accepted_length > verify_width:
            failures.append(
                _failure(
                    label,
                    "accepted_length_range",
                    f"observed {accepted_length}, expected [1, {verify_width}]",
                )
            )

        pair = (slot_id, accepted_length)
        if mode == "capture":
            if not graph_warmup:
                failures.append(
                    _failure(
                        label,
                        "runtime_capture",
                        f"capture observed outside startup warmup for {pair}",
                    )
                )
            else:
                capture_counts[pair] += 1
                capture_graph_keys.setdefault(pair, set()).add(graph_key)
        elif mode == "replay":
            if not bool(record["static_graph_tasks_prepared"]):
                failures.append(
                    _failure(
                        label,
                        "static_task_signal",
                        f"replay for {pair} did not pre-signal static tasks",
                    )
                )
            if graph_warmup:
                failures.append(
                    _failure(
                        label,
                        "warmup_replay",
                        f"warmup replayed an already captured variant for {pair}",
                    )
                )
            else:
                runtime_replay_counts[pair] += 1
                runtime_replay_graph_keys[(slot_id, accepted_length, graph_key)] += 1

    expected_pairs = {
        (slot_id, accepted_length) for slot_id in range(slot_count) for accepted_length in range(1, verify_width + 1)
    }
    missing_pairs = sorted(expected_pairs - set(capture_counts))
    duplicate_pairs = sorted(pair for pair, count in capture_counts.items() if count != 1)
    if missing_pairs:
        failures.append(
            _failure(
                label,
                "warmup_capture_coverage",
                f"missing Slot/accepted-length pairs: {missing_pairs}",
            )
        )
    if duplicate_pairs:
        failures.append(
            _failure(
                label,
                "warmup_capture_uniqueness",
                f"capture count is not exactly one for pairs: {duplicate_pairs}",
            )
        )

    replay_without_capture = sorted(pair for pair in runtime_replay_counts if pair not in capture_counts)
    if replay_without_capture:
        failures.append(
            _failure(
                label,
                "runtime_replay_provenance",
                f"runtime replay lacks a startup capture: {replay_without_capture}",
            )
        )
    replay_graph_key_mismatches = sorted(
        (slot_id, accepted_length, graph_key)
        for slot_id, accepted_length, graph_key in runtime_replay_graph_keys
        if graph_key not in capture_graph_keys.get((slot_id, accepted_length), set())
    )
    if replay_graph_key_mismatches:
        failures.append(
            _failure(
                label,
                "runtime_replay_graph_key",
                f"runtime replay Graph key was not captured during startup: {replay_graph_key_mismatches}",
            )
        )
    graph_key_accepted_lengths: dict[tuple[int, int], set[int]] = {}
    for (slot_id, accepted_length), graph_keys in capture_graph_keys.items():
        for graph_key in graph_keys:
            graph_key_accepted_lengths.setdefault((slot_id, graph_key), set()).add(accepted_length)
    graph_key_collisions = sorted(
        (slot_id, graph_key, sorted(accepted_lengths))
        for (slot_id, graph_key), accepted_lengths in (graph_key_accepted_lengths.items())
        if len(accepted_lengths) != 1
    )
    if graph_key_collisions:
        failures.append(
            _failure(
                label,
                "warmup_graph_key_identity",
                f"one startup Graph key represents multiple accepted lengths: {graph_key_collisions}",
            )
        )
    runtime_replay_count = sum(runtime_replay_counts.values())
    if require_runtime_replay and runtime_replay_count == 0:
        failures.append(
            _failure(
                label,
                "runtime_replay_observed",
                "no runtime speculative Prepared Graph replay was observed",
            )
        )

    summary = {
        "label": label,
        "path": str(path),
        "passed": not failures,
        "event_count": len(records),
        "mode_counts": dict(sorted(mode_counts.items())),
        "warmup_capture_pairs": [
            {
                "slot_id": pair[0],
                "accepted_length": pair[1],
                "count": count,
                "graph_keys": sorted(capture_graph_keys.get(pair, set())),
            }
            for pair, count in sorted(capture_counts.items())
        ],
        "runtime_replay_pairs": [
            {
                "slot_id": pair[0],
                "accepted_length": pair[1],
                "count": count,
                "graph_keys": sorted(
                    graph_key
                    for (
                        replay_slot_id,
                        replay_accepted_length,
                        graph_key,
                    ) in runtime_replay_graph_keys
                    if replay_slot_id == pair[0] and replay_accepted_length == pair[1]
                ),
            }
            for pair, count in sorted(runtime_replay_counts.items())
        ],
    }
    return summary, failures


def _parse_trace_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"trace must use LABEL=PATH syntax: {value}")
    label, path_value = value.split("=", maxsplit=1)
    if not label or not path_value:
        raise ValueError(f"trace must use non-empty LABEL=PATH syntax: {value}")
    return label, Path(path_value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Rank-labeled verbose trace; repeat for every rank.",
    )
    parser.add_argument("--verify-width", type=int, required=True)
    parser.add_argument("--slot-count", type=int, required=True)
    parser.add_argument("--require-runtime-replay", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failed-cases", type=Path, required=True)
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    try:
        traces = [_parse_trace_argument(value) for value in args.trace]
        for label, path in traces:
            summary, trace_failures = validate_prepared_speculative_graph_trace(
                path,
                label,
                args.verify_width,
                args.slot_count,
                args.require_runtime_replay,
            )
            summaries.append(summary)
            failures.extend(trace_failures)
    except (OSError, ValueError):
        logger.exception("failed to validate speculative Prepared Graph trace")
        return 2

    report = {
        "passed": not failures,
        "verify_width": args.verify_width,
        "slot_count": args.slot_count,
        "traces": summaries,
        "failures": failures,
    }
    _write_json(args.output, report)
    _write_json(args.failed_cases, failures)
    if failures:
        logger.error(f"speculative Prepared Graph trace validation failed: {len(failures)} checks")
        return 1
    logger.info(f"speculative Prepared Graph trace validation passed: {len(summaries)} traces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
