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

import tempfile
import unittest
from pathlib import Path

from tools.compare_speculative_step_trace import (
    compare_speculative_step_trace_pairs,
    compare_speculative_step_traces,
    parse_speculative_step_trace,
)


def _record(
    request_id: str,
    embedding_id: int,
    base_position: int,
    base_kv_seq_len: int,
    tokens: str,
    algorithm: str = "DFlash",
) -> str:
    token_values = [int(token) for token in tokens.split(",")]
    committed_length = 0
    for token in token_values:
        if token < 0:
            break
        committed_length += 1
    accepted_draft_length = max(committed_length - 1, 0)
    return (
        "2026-08-25 21:00:00.000 event=speculative_step_state "
        f"algorithm={algorithm} request_id={request_id} "
        f"embedding_id={embedding_id} base_position={base_position} "
        f"base_kv_seq_len={base_kv_seq_len} "
        f"committed_length={committed_length} "
        f"accepted_draft_length={accepted_draft_length} tokens={tokens}\n"
    )


class CompareSpeculativeStepTraceTest(unittest.TestCase):
    def _write(self, directory: Path, name: str, content: str) -> Path:
        path = directory / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_exact_state_sequences_match_per_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            content = (
                _record("request-a", 1, 8, 9, "11,12,-1")
                + _record("request-b", 2, 15, 16, "21,-1,-1")
                + _record("request-a", 1, 10, 11, "13,14,15")
            )
            baseline = self._write(directory, "baseline.log", content)
            candidate = self._write(directory, "candidate.log", content)

            summary, failures = compare_speculative_step_traces(
                baseline,
                candidate,
                "prepared",
                expected_algorithm="DFlash",
            )

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["compared_steps"], 3)
        self.assertEqual(failures, [])

    def test_token_or_state_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            baseline = self._write(
                directory,
                "baseline.log",
                _record("request-a", 1, 8, 9, "11,12,-1"),
            )
            candidate = self._write(
                directory,
                "candidate.log",
                _record("request-a", 1, 8, 10, "11,-1,-1"),
            )

            summary, failures = compare_speculative_step_traces(
                baseline,
                candidate,
                "prepared",
                expected_algorithm="DFlash",
            )

        self.assertFalse(summary["passed"])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["reason"], "step_state_mismatch")

    def test_multiple_rank_pairs_are_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            matching = _record("request-a", 1, 8, 9, "11,12,-1")
            mismatching = _record("request-a", 1, 9, 10, "12,-1,-1")
            baseline_rank_0 = self._write(directory, "b0.log", matching)
            candidate_rank_0 = self._write(directory, "c0.log", matching)
            baseline_rank_1 = self._write(directory, "b1.log", matching)
            candidate_rank_1 = self._write(directory, "c1.log", mismatching)

            result, failures = compare_speculative_step_trace_pairs(
                [
                    ("single-rank-0", baseline_rank_0, candidate_rank_0),
                    ("single-rank-1", baseline_rank_1, candidate_rank_1),
                ],
                expected_algorithm="DFlash",
            )

        self.assertFalse(result["passed"])
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["matching_steps"], 1)
        self.assertEqual(len(failures), 1)

    def test_unexpected_algorithm_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = self._write(
                Path(temp_dir),
                "invalid.log",
                _record(
                    "request-a",
                    1,
                    8,
                    9,
                    "11,12,-1",
                    algorithm="DSpark",
                ),
            )

            with self.assertRaisesRegex(ValueError, "unexpected speculative"):
                parse_speculative_step_trace(trace, expected_algorithm="DFlash")

    def test_non_suffix_padding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = self._write(
                Path(temp_dir),
                "invalid.log",
                (
                    "event=speculative_step_state algorithm=DFlash "
                    "request_id=request-a embedding_id=1 base_position=8 "
                    "base_kv_seq_len=9 committed_length=2 "
                    "accepted_draft_length=1 tokens=11,-1,12\n"
                ),
            )

            with self.assertRaisesRegex(ValueError, "follows a negative"):
                parse_speculative_step_trace(trace)


if __name__ == "__main__":
    unittest.main()
