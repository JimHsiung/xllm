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

from tools.compare_e2e_performance import compare_e2e_performance_pairs


def _result(
    duration_seconds: float,
    completion_tokens: int,
    p50_seconds: float,
    p95_seconds: float,
) -> dict[str, Any]:
    return {
        "model": "Qwen3-0.6B",
        "configuration": {"stress_rounds": 6, "parallel_requests": 4},
        "summary": {
            "total_requests": 27,
            "successful_requests": 27,
            "failed_requests": 0,
            "duration_seconds": duration_seconds,
            "prompt_tokens": 8000,
            "completion_tokens": completion_tokens,
            "total_tokens": 8000 + completion_tokens,
            "latency_seconds": {"p50": p50_seconds, "p95": p95_seconds},
        },
    }


class CompareE2EPerformanceTest(unittest.TestCase):
    def _write(self, directory: Path, name: str, result: dict[str, Any]) -> Path:
        path = directory / name
        path.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def test_improvement_passes_all_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            baseline = self._write(directory, "baseline.json", _result(10.0, 1000, 2.0, 3.0))
            candidate = self._write(directory, "candidate.json", _result(8.0, 1000, 1.5, 2.5))

            result = compare_e2e_performance_pairs(
                [("round-1", baseline, candidate)],
                min_total_token_throughput_ratio=1.1,
                min_completion_token_throughput_ratio=1.1,
                max_p50_latency_ratio=0.9,
                max_p95_latency_ratio=0.9,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])

    def test_regression_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            baseline = self._write(directory, "baseline.json", _result(8.0, 1000, 1.5, 2.5))
            candidate = self._write(directory, "candidate.json", _result(10.0, 1000, 2.0, 3.0))

            result = compare_e2e_performance_pairs(
                [("round-1", baseline, candidate)],
                min_total_token_throughput_ratio=1.0,
                min_completion_token_throughput_ratio=1.0,
                max_p50_latency_ratio=1.0,
                max_p95_latency_ratio=1.0,
            )

        self.assertFalse(result["passed"])
        self.assertEqual(
            set(result["failures"]),
            {
                "total_token_throughput",
                "completion_token_throughput",
                "p50_latency",
                "p95_latency",
            },
        )

    def test_median_of_paired_rounds_controls_one_outlier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            pairs = []
            for index, candidate_duration in enumerate((8.0, 8.0, 20.0)):
                baseline = self._write(
                    directory,
                    f"baseline-{index}.json",
                    _result(10.0, 1000, 2.0, 3.0),
                )
                candidate = self._write(
                    directory,
                    f"candidate-{index}.json",
                    _result(candidate_duration, 1000, 1.5, 2.5),
                )
                pairs.append((f"round-{index}", baseline, candidate))

            result = compare_e2e_performance_pairs(pairs, min_total_token_throughput_ratio=1.1)

        self.assertTrue(result["passed"])
        self.assertEqual(result["median_ratios"]["total_token_throughput_ratio"], 1.25)

    def test_display_rounding_does_not_mask_threshold_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            baseline = self._write(directory, "baseline.json", _result(10.0, 1000, 2.0, 3.0))
            candidate = self._write(
                directory,
                "candidate.json",
                _result(10.0 / 0.9499996, 1000, 2.0, 3.0),
            )

            result = compare_e2e_performance_pairs(
                [("round-1", baseline, candidate)],
                min_total_token_throughput_ratio=0.95,
            )

        self.assertEqual(result["median_ratios"]["total_token_throughput_ratio"], 0.95)
        self.assertFalse(result["passed"])

    def test_model_or_configuration_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            baseline_result = _result(10.0, 1000, 2.0, 3.0)
            candidate_result = _result(8.0, 1000, 1.5, 2.5)
            candidate_result["configuration"] = {"stress_rounds": 7}
            baseline = self._write(directory, "baseline.json", baseline_result)
            candidate = self._write(directory, "candidate.json", candidate_result)

            with self.assertRaisesRegex(ValueError, "same model"):
                compare_e2e_performance_pairs([("round-1", baseline, candidate)])

    def test_failed_requests_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            baseline_result = _result(10.0, 1000, 2.0, 3.0)
            baseline_result["summary"]["successful_requests"] = 26
            baseline_result["summary"]["failed_requests"] = 1
            baseline = self._write(directory, "baseline.json", baseline_result)
            candidate = self._write(
                directory,
                "candidate.json",
                _result(8.0, 1000, 1.5, 2.5),
            )

            with self.assertRaisesRegex(ValueError, "failed requests"):
                compare_e2e_performance_pairs([("round-1", baseline, candidate)])

    def test_token_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            baseline = self._write(directory, "baseline.json", _result(10.0, 1000, 2.0, 3.0))
            candidate = self._write(directory, "candidate.json", _result(8.0, 999, 1.5, 2.5))

            with self.assertRaisesRegex(ValueError, "equal token counts"):
                compare_e2e_performance_pairs([("round-1", baseline, candidate)])


if __name__ == "__main__":
    unittest.main()
