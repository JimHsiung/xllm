# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

import copy
import unittest
from typing import Any

from tools.validate_prepared_task_trace import validate_prepared_task_trace


def _report() -> dict[str, Any]:
    marker = {"count": 12}
    return {
        "schema_version": 1,
        "markers": {
            "xllm.PreparedTask.Prepare": dict(marker),
            "xllm.PreparedTask.Launch": dict(marker),
            "xllm.PreparedTask.Consume": dict(marker),
        },
        "copies": {
            "host to device": {
                "workload_count": 12,
                "workload_bytes": 4096,
                "workload_max_bytes": 512,
            },
            "device to device": {
                "workload_count": 2,
                "workload_bytes": 1024,
                "workload_max_bytes": 512,
            },
        },
        "allocations": {
            "aclrtMallocPhysical": {"workload_count": 0},
        },
        "metrics": {
            "delta": {
                "prepared_task_execution_total_graph_replay": 9,
                "prepared_task_graph_address_mismatches_total": 0,
            }
        },
        "consistency_checks": {
            "marker_counts_equal": True,
            "metrics_available": True,
            "prepare_markers_match_metric_delta": True,
            "launch_markers_match_metric_delta": True,
            "consume_markers_match_metric_delta": True,
            "h2d_count_matches_metric_delta": True,
            "h2d_bytes_match_metric_delta": True,
            "prepared_d2d_metric_is_zero": True,
        },
        "prepare_temporal_prior_launch_device_overlap": {
            "matched_prepare_count": 11,
            "overlapping_prepare_count": 10,
        },
    }


def _validate(report: dict[str, Any], **overrides: bool) -> dict[str, Any]:
    options = {
        "require_zero_cann_allocation": True,
        "allow_prepared_d2d": False,
        "require_prepare_device_overlap": True,
        "require_graph_replay": True,
        "require_graph_address_stability": True,
    }
    options.update(overrides)
    return validate_prepared_task_trace(report, arena_capacity_bytes=4096, **options)


class ValidatePreparedTaskTraceTest(unittest.TestCase):
    def test_complete_graph_trace_passes(self) -> None:
        result = _validate(_report())

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])

    def test_marker_mismatch_fails(self) -> None:
        report = _report()
        report["markers"]["xllm.PreparedTask.Consume"]["count"] = 11

        result = _validate(report)

        self.assertFalse(result["passed"])
        self.assertIn("prepared_marker_counts_equal", result["failures"])

    def test_warmed_allocation_fails(self) -> None:
        report = _report()
        report["allocations"]["aclrtMallocPhysical"]["workload_count"] = 1

        result = _validate(report)

        self.assertFalse(result["passed"])
        self.assertIn("warmed_cann_allocation_is_zero", result["failures"])

    def test_full_arena_d2d_fails_even_when_staging_d2d_is_allowed(self) -> None:
        report = _report()
        report["copies"]["device to device"]["workload_max_bytes"] = 4096
        report["consistency_checks"]["prepared_d2d_metric_is_zero"] = False

        result = _validate(report, allow_prepared_d2d=True)

        self.assertFalse(result["passed"])
        self.assertIn("no_full_arena_d2d", result["failures"])
        self.assertNotIn("prepared_staging_d2d_is_zero", result["checks"])

    def test_graph_replay_and_address_mismatch_are_gated(self) -> None:
        report = _report()
        report["metrics"]["delta"]["prepared_task_execution_total_graph_replay"] = 0
        report["metrics"]["delta"]["prepared_task_graph_address_mismatches_total"] = 1

        result = _validate(report)

        self.assertFalse(result["passed"])
        self.assertIn("graph_replay_observed", result["failures"])
        self.assertIn("graph_address_mismatches_are_zero", result["failures"])

    def test_overlap_requirement_is_optional(self) -> None:
        report = _report()
        report["prepare_temporal_prior_launch_device_overlap"]["overlapping_prepare_count"] = 0

        failed = _validate(report)
        passed = _validate(report, require_prepare_device_overlap=False)

        self.assertFalse(failed["passed"])
        self.assertTrue(passed["passed"])

    def test_does_not_mutate_report(self) -> None:
        report = _report()
        original = copy.deepcopy(report)

        _validate(report)

        self.assertEqual(report, original)


if __name__ == "__main__":
    unittest.main()
