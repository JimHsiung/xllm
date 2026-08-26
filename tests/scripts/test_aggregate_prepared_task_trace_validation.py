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

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tools.aggregate_prepared_task_trace_validation import (
    aggregate_prepared_task_trace_validations,
)


def _report(passed: bool, prepare_count: int, d2d_bytes: int) -> dict[str, Any]:
    failures = [] if passed else ["graph_replay_observed"]
    return {
        "passed": passed,
        "marker_counts": {
            "xllm.PreparedTask.Prepare": prepare_count,
            "xllm.PreparedTask.Launch": prepare_count,
            "xllm.PreparedTask.Consume": prepare_count,
        },
        "maximum_device_to_device_copy_bytes": d2d_bytes,
        "failures": failures,
    }


class AggregatePreparedTaskTraceValidationTest(unittest.TestCase):
    def _write(
        self,
        directory: Path,
        rank: int,
        report: dict[str, Any],
    ) -> Path:
        path = directory / f"rank-{rank}.json"
        path.write_text(
            json.dumps(report, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def test_two_passing_ranks_are_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            rank_zero = self._write(directory, 0, _report(True, 10, 1024))
            rank_one = self._write(directory, 1, _report(True, 12, 2048))

            result = aggregate_prepared_task_trace_validations([(0, rank_zero), (1, rank_one)])

        self.assertTrue(result["passed"])
        self.assertEqual(result["rank_count"], 2)
        self.assertEqual(
            result["aggregate_marker_counts"]["xllm.PreparedTask.Prepare"],
            22,
        )
        self.assertEqual(result["maximum_device_to_device_copy_bytes"], 2048)

    def test_failing_rank_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            rank_zero = self._write(directory, 0, _report(True, 10, 1024))
            rank_one = self._write(directory, 1, _report(False, 10, 1024))

            result = aggregate_prepared_task_trace_validations([(0, rank_zero), (1, rank_one)])

        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["rank"], 1)

    def test_duplicate_or_noncontiguous_rank_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            report = self._write(directory, 0, _report(True, 10, 1024))

            with self.assertRaisesRegex(ValueError, "duplicate"):
                aggregate_prepared_task_trace_validations([(0, report), (0, report)])
            with self.assertRaisesRegex(ValueError, "contiguous"):
                aggregate_prepared_task_trace_validations([(1, report)])

    def test_malformed_rank_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = directory / "rank-0.json"
            path.write_text('{"passed": true}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "marker counts"):
                aggregate_prepared_task_trace_validations([(0, path)])


if __name__ == "__main__":
    unittest.main()
