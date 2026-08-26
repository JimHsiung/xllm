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
"""Compare deterministic MTP/Eagle3 step-state verbose trace records."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

_STEP_PATTERN = re.compile(
    r"\bevent=mtp_step_state "
    r"(?:algorithm=(?P<algorithm>\S+) )?"
    r"request_id=(?P<request_id>\S+) "
    r"embedding_id=(?P<embedding_id>-?\d+) "
    r"base_position=(?P<base_position>-?\d+) "
    r"base_kv_seq_len=(?P<base_kv_seq_len>-?\d+) "
    r"committed_length=(?P<committed_length>\d+) "
    r"accepted_draft_length=(?P<accepted_draft_length>\d+) "
    r"tokens=(?P<tokens>-?\d+(?:,-?\d+)*)\s*$"
)


def _parse_step_line(line: str, path: Path, line_number: int) -> dict[str, Any]:
    match = _STEP_PATTERN.search(line)
    if match is None:
        raise ValueError(f"malformed mtp_step_state record: {path}:{line_number}: {line.rstrip()}")

    tokens = [int(token) for token in match.group("tokens").split(",")]
    committed_length = 0
    saw_padding = False
    for token in tokens:
        if token < 0:
            saw_padding = True
            continue
        if saw_padding:
            raise ValueError(f"non-padding MTP token follows a negative padding token: {path}:{line_number}")
        committed_length += 1

    recorded_committed_length = int(match.group("committed_length"))
    if recorded_committed_length != committed_length:
        raise ValueError(f"MTP committed length does not match token padding: {path}:{line_number}")
    recorded_accepted_draft_length = int(match.group("accepted_draft_length"))
    if recorded_accepted_draft_length != max(committed_length - 1, 0):
        raise ValueError(f"MTP accepted draft length does not match committed tokens: {path}:{line_number}")

    return {
        "algorithm": match.group("algorithm") or "MTP",
        "request_id": match.group("request_id"),
        "embedding_id": int(match.group("embedding_id")),
        "base_position": int(match.group("base_position")),
        "base_kv_seq_len": int(match.group("base_kv_seq_len")),
        "committed_length": recorded_committed_length,
        "accepted_draft_length": recorded_accepted_draft_length,
        "tokens": tokens,
    }


def parse_mtp_step_trace(
    path: Path,
    require_request_ids: bool = True,
    require_base_state: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Parse step records, retaining each request's original step order."""
    records: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            if "event=mtp_step_state" not in line:
                continue
            record = _parse_step_line(line, path, line_number)
            request_id = str(record.pop("request_id"))
            if require_request_ids and request_id == "-":
                raise ValueError(f"MTP step record has no request id: {path}:{line_number}")
            if require_base_state and (int(record["base_position"]) < 0 or int(record["base_kv_seq_len"]) < 0):
                raise ValueError(f"MTP step record has no base state: {path}:{line_number}")
            records.setdefault(request_id, []).append(record)

    if not records:
        raise ValueError(f"no mtp_step_state records found in {path}")
    return records


def compare_mtp_step_traces(
    baseline_path: Path,
    candidate_path: Path,
    label: str,
    require_request_ids: bool = True,
    require_base_state: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare exact per-request state sequences from two trace files."""
    baseline = parse_mtp_step_trace(
        baseline_path,
        require_request_ids=require_request_ids,
        require_base_state=require_base_state,
    )
    candidate = parse_mtp_step_trace(
        candidate_path,
        require_request_ids=require_request_ids,
        require_base_state=require_base_state,
    )
    failures: list[dict[str, Any]] = []
    baseline_request_ids = set(baseline)
    candidate_request_ids = set(candidate)
    for request_id in sorted(baseline_request_ids - candidate_request_ids):
        failures.append(
            {
                "label": label,
                "request_id": request_id,
                "reason": "missing_candidate_request",
            }
        )
    for request_id in sorted(candidate_request_ids - baseline_request_ids):
        failures.append(
            {
                "label": label,
                "request_id": request_id,
                "reason": "unexpected_candidate_request",
            }
        )

    compared_steps = 0
    matching_steps = 0
    for request_id in sorted(baseline_request_ids & candidate_request_ids):
        baseline_steps = baseline[request_id]
        candidate_steps = candidate[request_id]
        if len(baseline_steps) != len(candidate_steps):
            failures.append(
                {
                    "label": label,
                    "request_id": request_id,
                    "reason": "step_count_mismatch",
                    "baseline_step_count": len(baseline_steps),
                    "candidate_step_count": len(candidate_steps),
                }
            )
        shared_step_count = min(len(baseline_steps), len(candidate_steps))
        compared_steps += shared_step_count
        for step_index in range(shared_step_count):
            baseline_step = baseline_steps[step_index]
            candidate_step = candidate_steps[step_index]
            if baseline_step == candidate_step:
                matching_steps += 1
                continue
            failures.append(
                {
                    "label": label,
                    "request_id": request_id,
                    "step_index": step_index,
                    "reason": "step_state_mismatch",
                    "baseline": baseline_step,
                    "candidate": candidate_step,
                }
            )

    summary = {
        "label": label,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "baseline_requests": len(baseline),
        "candidate_requests": len(candidate),
        "compared_steps": compared_steps,
        "matching_steps": matching_steps,
        "mismatching_steps": compared_steps - matching_steps,
        "passed": not failures,
    }
    return summary, failures


def compare_mtp_step_trace_pairs(
    pairs: list[tuple[str, Path, Path]],
    require_request_ids: bool = True,
    require_base_state: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare and aggregate independently paired MTP rank traces."""
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for label, baseline_path, candidate_path in pairs:
        summary, pair_failures = compare_mtp_step_traces(
            baseline_path,
            candidate_path,
            label,
            require_request_ids=require_request_ids,
            require_base_state=require_base_state,
        )
        summaries.append(summary)
        failures.extend(pair_failures)

    result = {
        "passed": not failures,
        "pair_count": len(summaries),
        "compared_steps": sum(int(summary["compared_steps"]) for summary in summaries),
        "matching_steps": sum(int(summary["matching_steps"]) for summary in summaries),
        "pairs": summaries,
    }
    return result, failures


def _parse_candidate(candidate_spec: str) -> tuple[str, Path]:
    if "=" not in candidate_spec:
        raise ValueError("candidate must use LABEL=TRACE_PATH syntax")
    label, path = candidate_spec.split("=", 1)
    if not label or not path:
        raise ValueError("candidate label and trace path must be non-empty")
    return label, Path(path)


def _parse_pair(pair_spec: str) -> tuple[str, Path, Path]:
    if "=" not in pair_spec:
        raise ValueError("pair must use LABEL=BASELINE_PATH,CANDIDATE_PATH syntax")
    label, paths = pair_spec.split("=", 1)
    if "," not in paths:
        raise ValueError("pair must use LABEL=BASELINE_PATH,CANDIDATE_PATH syntax")
    baseline_path, candidate_path = paths.split(",", 1)
    if not label or not baseline_path or not candidate_path:
        raise ValueError("pair label and trace paths must be non-empty")
    return label, Path(baseline_path), Path(candidate_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        help="LABEL=TRACE_PATH; may be supplied more than once",
    )
    parser.add_argument(
        "--pair",
        action="append",
        help=("LABEL=BASELINE_PATH,CANDIDATE_PATH; may be supplied more than once for corresponding rank traces"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--failed-cases", required=True, type=Path)
    parser.add_argument(
        "--allow-missing-request-ids",
        action="store_true",
        help="allow '-' request ids for non-service unit traces",
    )
    parser.add_argument(
        "--allow-missing-base-state",
        action="store_true",
        help="allow -1 base position/KV state for adaptive legacy traces",
    )
    args = parser.parse_args()

    candidate_specs = args.candidate or []
    pair_specs = args.pair or []
    if pair_specs and (args.baseline is not None or candidate_specs):
        parser.error("--pair cannot be combined with --baseline/--candidate")
    if not pair_specs and (args.baseline is None or not candidate_specs):
        parser.error("provide --pair or both --baseline and --candidate")

    pairs: list[tuple[str, Path, Path]] = []
    if pair_specs:
        pairs.extend(_parse_pair(pair_spec) for pair_spec in pair_specs)
    else:
        assert args.baseline is not None
        for candidate_spec in candidate_specs:
            label, candidate_path = _parse_candidate(candidate_spec)
            pairs.append((label, args.baseline, candidate_path))

    result, failures = compare_mtp_step_trace_pairs(
        pairs,
        require_request_ids=not args.allow_missing_request_ids,
        require_base_state=not args.allow_missing_base_state,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.failed_cases.write_text(
        "".join(json.dumps(failure, ensure_ascii=False) + "\n" for failure in failures),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} MTP step-state comparisons failed")


if __name__ == "__main__":
    main()
