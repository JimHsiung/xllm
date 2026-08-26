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

import tempfile
import unittest
from pathlib import Path

from tools.validate_prepared_speculative_graph_trace import (
    parse_prepared_speculative_graph_trace,
    validate_prepared_speculative_graph_trace,
)


def _record(
    mode: str,
    slot_id: int,
    accepted_length: int,
    verify_width: int,
    graph_warmup: bool,
    static_graph_tasks_prepared: bool,
    graph_key: int | None = None,
) -> str:
    if graph_key is None:
        graph_key = 1000 + slot_id * 100 + accepted_length
    return (
        "2026-08-26 06:00:00.000 "
        f"event=prepared_speculative_graph_execution mode={mode} "
        f"slot_id={slot_id} graph_key={graph_key} "
        f"graph_warmup={int(graph_warmup)} "
        f"accepted_length={accepted_length} verify_width={verify_width} "
        "static_graph_tasks_prepared="
        f"{int(static_graph_tasks_prepared)}\n"
    )


def _capture_records(verify_width: int, slot_count: int) -> str:
    return "".join(
        _record(
            "capture",
            slot_id,
            accepted_length,
            verify_width,
            graph_warmup=True,
            static_graph_tasks_prepared=False,
        )
        for accepted_length in range(1, verify_width + 1)
        for slot_id in range(slot_count)
    )


class ValidatePreparedSpeculativeGraphTraceTest(unittest.TestCase):
    def _write(self, directory: Path, content: str) -> Path:
        path = directory / "trace.log"
        path.write_text(content, encoding="utf-8")
        return path

    def test_all_slot_and_accepted_length_variants_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            trace = self._write(
                directory,
                _capture_records(verify_width=4, slot_count=2)
                + _record(
                    "replay",
                    slot_id=1,
                    accepted_length=3,
                    verify_width=4,
                    graph_warmup=False,
                    static_graph_tasks_prepared=True,
                )
                + _record(
                    "eager",
                    slot_id=0,
                    accepted_length=2,
                    verify_width=4,
                    graph_warmup=False,
                    static_graph_tasks_prepared=False,
                ),
            )
            summary, failures = validate_prepared_speculative_graph_trace(
                trace,
                "rank-0",
                verify_width=4,
                slot_count=2,
                require_runtime_replay=True,
            )

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["mode_counts"]["capture"], 8)
        self.assertEqual(summary["mode_counts"]["replay"], 1)
        self.assertEqual(failures, [])

    def test_missing_warmup_variant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            content = _capture_records(verify_width=3, slot_count=2)
            missing_record = _record(
                "capture",
                slot_id=1,
                accepted_length=3,
                verify_width=3,
                graph_warmup=True,
                static_graph_tasks_prepared=False,
            )
            trace = self._write(directory, content.replace(missing_record, ""))
            summary, failures = validate_prepared_speculative_graph_trace(trace, "rank-1", verify_width=3, slot_count=2)

        self.assertFalse(summary["passed"])
        self.assertIn(
            "warmup_capture_coverage",
            {failure["check"] for failure in failures},
        )

    def test_duplicate_warmup_variant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            duplicate = _record(
                "capture",
                slot_id=0,
                accepted_length=1,
                verify_width=2,
                graph_warmup=True,
                static_graph_tasks_prepared=False,
            )
            trace = self._write(
                directory,
                _capture_records(verify_width=2, slot_count=1) + duplicate,
            )
            summary, failures = validate_prepared_speculative_graph_trace(trace, "rank-0", verify_width=2, slot_count=1)

        self.assertFalse(summary["passed"])
        self.assertIn(
            "warmup_capture_uniqueness",
            {failure["check"] for failure in failures},
        )

    def test_runtime_replay_requires_static_task_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            trace = self._write(
                directory,
                _capture_records(verify_width=2, slot_count=1)
                + _record(
                    "replay",
                    slot_id=0,
                    accepted_length=2,
                    verify_width=2,
                    graph_warmup=False,
                    static_graph_tasks_prepared=False,
                ),
            )
            summary, failures = validate_prepared_speculative_graph_trace(
                trace,
                "rank-0",
                verify_width=2,
                slot_count=1,
                require_runtime_replay=True,
            )

        self.assertFalse(summary["passed"])
        self.assertIn("static_task_signal", {failure["check"] for failure in failures})

    def test_runtime_replay_requires_captured_graph_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            trace = self._write(
                directory,
                _capture_records(verify_width=2, slot_count=1)
                + _record(
                    "replay",
                    slot_id=0,
                    accepted_length=2,
                    verify_width=2,
                    graph_warmup=False,
                    static_graph_tasks_prepared=True,
                    graph_key=9999,
                ),
            )
            summary, failures = validate_prepared_speculative_graph_trace(
                trace,
                "rank-0",
                verify_width=2,
                slot_count=1,
                require_runtime_replay=True,
            )

        self.assertFalse(summary["passed"])
        self.assertIn(
            "runtime_replay_graph_key",
            {failure["check"] for failure in failures},
        )

    def test_accepted_lengths_require_distinct_graph_keys_per_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            trace = self._write(
                directory,
                _record(
                    "capture",
                    slot_id=0,
                    accepted_length=1,
                    verify_width=2,
                    graph_warmup=True,
                    static_graph_tasks_prepared=False,
                    graph_key=1001,
                )
                + _record(
                    "capture",
                    slot_id=0,
                    accepted_length=2,
                    verify_width=2,
                    graph_warmup=True,
                    static_graph_tasks_prepared=False,
                    graph_key=1001,
                ),
            )
            summary, failures = validate_prepared_speculative_graph_trace(trace, "rank-0", verify_width=2, slot_count=1)

        self.assertFalse(summary["passed"])
        self.assertIn(
            "warmup_graph_key_identity",
            {failure["check"] for failure in failures},
        )

    def test_invalid_dimensions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            trace = self._write(
                directory,
                _capture_records(verify_width=2, slot_count=1)
                + _record(
                    "eager",
                    slot_id=2,
                    accepted_length=0,
                    verify_width=3,
                    graph_warmup=False,
                    static_graph_tasks_prepared=False,
                ),
            )
            summary, failures = validate_prepared_speculative_graph_trace(trace, "rank-0", verify_width=2, slot_count=1)

        self.assertFalse(summary["passed"])
        checks = {failure["check"] for failure in failures}
        self.assertIn("verify_width", checks)
        self.assertIn("slot_range", checks)
        self.assertIn("accepted_length_range", checks)

    def test_warmup_replay_without_capture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            trace = self._write(
                directory,
                _record(
                    "capture",
                    slot_id=0,
                    accepted_length=1,
                    verify_width=2,
                    graph_warmup=True,
                    static_graph_tasks_prepared=False,
                )
                + _record(
                    "replay",
                    slot_id=0,
                    accepted_length=1,
                    verify_width=2,
                    graph_warmup=True,
                    static_graph_tasks_prepared=True,
                )
                + _record(
                    "replay",
                    slot_id=0,
                    accepted_length=2,
                    verify_width=2,
                    graph_warmup=False,
                    static_graph_tasks_prepared=True,
                ),
            )
            summary, failures = validate_prepared_speculative_graph_trace(trace, "rank-0", verify_width=2, slot_count=1)

        self.assertFalse(summary["passed"])
        checks = {failure["check"] for failure in failures}
        self.assertIn("warmup_replay", checks)
        self.assertIn("runtime_replay_provenance", checks)

    def test_runtime_capture_and_missing_runtime_replay_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            trace = self._write(
                directory,
                _capture_records(verify_width=2, slot_count=1)
                + _record(
                    "capture",
                    slot_id=0,
                    accepted_length=1,
                    verify_width=2,
                    graph_warmup=False,
                    static_graph_tasks_prepared=False,
                ),
            )
            summary, failures = validate_prepared_speculative_graph_trace(
                trace,
                "rank-0",
                verify_width=2,
                slot_count=1,
                require_runtime_replay=True,
            )

        self.assertFalse(summary["passed"])
        checks = {failure["check"] for failure in failures}
        self.assertIn("runtime_capture", checks)
        self.assertIn("runtime_replay_observed", checks)

    def test_malformed_event_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = self._write(
                Path(temp_dir),
                "event=prepared_speculative_graph_execution mode=replay\n",
            )
            with self.assertRaisesRegex(ValueError, "malformed"):
                parse_prepared_speculative_graph_trace(trace)


if __name__ == "__main__":
    unittest.main()
