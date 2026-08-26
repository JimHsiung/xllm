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

from tools.compare_llm_step_trace import compare_llm_step_trace_pairs, compare_llm_step_traces, parse_llm_step_trace


def _record(
    request_id: str,
    embedding_id: int,
    base_position: int,
    base_kv_seq_len: int,
    token: int,
) -> str:
    return (
        "2026-08-25 22:00:00.000 event=llm_step_state "
        f"request_id={request_id} embedding_id={embedding_id} "
        f"base_position={base_position} base_kv_seq_len={base_kv_seq_len} "
        f"token={token}\n"
    )


class CompareLlmStepTraceTest(unittest.TestCase):
    def _write(self, directory: Path, name: str, content: str) -> Path:
        path = directory / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_exact_state_sequences_match_per_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            content = (
                _record("request-a", 1, 8, 9, 11)
                + _record("request-b", 2, 15, 16, 21)
                + _record("request-a", 1, 9, 10, 12)
            )
            baseline = self._write(directory, "baseline.log", content)
            candidate = self._write(directory, "candidate.log", content)

            summary, failures = compare_llm_step_traces(baseline, candidate, "prepared-graph")

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["compared_steps"], 3)
        self.assertEqual(failures, [])

    def test_token_or_state_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            baseline = self._write(directory, "baseline.log", _record("request-a", 1, 8, 9, 11))
            candidate = self._write(directory, "candidate.log", _record("request-a", 1, 8, 10, 12))

            summary, failures = compare_llm_step_traces(baseline, candidate, "prepared-graph")

        self.assertFalse(summary["passed"])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["reason"], "step_state_mismatch")

    def test_multiple_pairs_are_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            matching = _record("request-a", 1, 8, 9, 11)
            mismatch = _record("request-a", 1, 9, 10, 12)
            baseline_a = self._write(directory, "ba.log", matching)
            candidate_a = self._write(directory, "ca.log", matching)
            baseline_b = self._write(directory, "bb.log", matching)
            candidate_b = self._write(directory, "cb.log", mismatch)

            result, failures = compare_llm_step_trace_pairs(
                [
                    ("single", baseline_a, candidate_a),
                    ("double", baseline_b, candidate_b),
                ]
            )

        self.assertFalse(result["passed"])
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["matching_steps"], 1)
        self.assertEqual(len(failures), 1)

    def test_missing_request_or_fake_token_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            missing_request = self._write(directory, "missing.log", _record("-", 1, 8, 9, 11))
            fake_token = self._write(directory, "fake.log", _record("request-a", 1, 8, 9, -1))

            with self.assertRaisesRegex(ValueError, "no request id"):
                parse_llm_step_trace(missing_request)
            with self.assertRaisesRegex(ValueError, "fake token"):
                parse_llm_step_trace(fake_token)


if __name__ == "__main__":
    unittest.main()
