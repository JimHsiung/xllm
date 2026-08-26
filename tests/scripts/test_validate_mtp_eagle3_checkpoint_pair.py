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

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tools.validate_mtp_eagle3_checkpoint_pair import (
    validate_mtp_eagle3_checkpoint_pair,
)


def _mtp_target_config() -> dict[str, Any]:
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1,
        "vocab_size": 129280,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }


def _mtp_draft_config() -> dict[str, Any]:
    return {
        "architectures": ["DeepseekV4MtpForCausalLM"],
        "model_type": "deepseek_v4_mtp",
        "hidden_size": 4096,
        "num_hidden_layers": 1,
        "num_nextn_predict_layers": 1,
        "vocab_size": 129280,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }


def _eagle3_target_config() -> dict[str, Any]:
    return {
        "architectures": ["KimiK25ForConditionalGeneration"],
        "model_type": "kimi_k25",
        "bos_token_id": 163584,
        "eos_token_id": 163585,
        "text_config": {
            "model_type": "kimi_k2",
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "vocab_size": 163840,
        },
    }


def _eagle3_draft_config() -> dict[str, Any]:
    return {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "kimi_k25_eagle3",
        "hidden_size": 7168,
        "num_hidden_layers": 1,
        "vocab_size": 163840,
        "draft_vocab_size": 163840,
        "bos_token_id": 163584,
        "eos_token_id": 163585,
    }


def _fake_weight_shapes(
    config: dict[str, Any],
) -> dict[str, list[int]] | None:
    model_type = config.get("model_type")
    if model_type == "deepseek_v4":
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        last_layer = config["num_hidden_layers"] - 1
        hc_mult = config.get("hc_mult", 4)
        return {
            "embed.weight": [vocab_size, hidden_size],
            "layers.0.attn_norm.weight": [hidden_size],
            f"layers.{last_layer}.attn_norm.weight": [hidden_size],
            "norm.weight": [hidden_size],
            "head.weight": [vocab_size, hidden_size],
            "hc_head_fn": [hc_mult, hc_mult * hidden_size],
            "hc_head_base": [hc_mult],
            "hc_head_scale": [1],
        }
    if model_type == "deepseek_v4_mtp":
        hidden_size = config["hidden_size"]
        hc_mult = config.get("hc_mult", 4)
        prefix = "model.layers.0."
        return {
            prefix + "enorm.weight": [hidden_size],
            prefix + "hnorm.weight": [hidden_size],
            prefix + "e_proj.weight": [hidden_size, hidden_size],
            prefix + "h_proj.weight": [hidden_size, hidden_size],
            prefix + "norm.weight": [hidden_size],
            prefix + "attn_norm.weight": [hidden_size],
            prefix + "ffn_norm.weight": [hidden_size],
            prefix + "attn.wq_a.weight": [1024, hidden_size],
            prefix + "ffn.gate.weight": [256, hidden_size],
            prefix + "hc_head_fn": [hc_mult, hc_mult * hidden_size],
            prefix + "hc_head_base": [hc_mult],
            prefix + "hc_head_scale": [1],
        }
    if model_type == "kimi_k25":
        return {
            "language_model.model.embed_tokens.weight": [163840, 7168],
            "language_model.model.layers.0.input_layernorm.weight": [7168],
            "language_model.model.norm.weight": [7168],
            "language_model.lm_head.weight": [163840, 7168],
        }
    if model_type == "kimi_k25_eagle3":
        effective_vocab_size = config.get("draft_vocab_size", config["vocab_size"])
        return {
            "fc.weight": [7168, 21504],
            "midlayer.input_layernorm.weight": [7168],
            "norm.weight": [7168],
            "lm_head.weight": [effective_vocab_size, 7168],
        }
    return None


def _write_fake_safetensors(path: Path, weight_shapes: dict[str, list[int]]) -> None:
    header = {
        key: {
            "dtype": "F16",
            "shape": shape,
            "data_offsets": [0, 0],
        }
        for key, shape in weight_shapes.items()
    }
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes)


def _write_fake_safetensors_index(path: Path, weight_map: dict[str, str]) -> None:
    path.write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )


def _write_checkpoint(
    root: Path,
    name: str,
    config: dict[str, Any],
    with_weights: bool = True,
) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if with_weights:
        weight_path = directory / "model.safetensors"
        weight_shapes = _fake_weight_shapes(config)
        if weight_shapes is None:
            weight_path.touch()
        else:
            _write_fake_safetensors(weight_path, weight_shapes)
    return directory


class ValidateMtpEagle3CheckpointPairTest(unittest.TestCase):
    def _validate(
        self,
        target_config: dict[str, Any],
        draft_config: dict[str, Any],
        algorithm: str,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", target_config)
            draft = _write_checkpoint(root, "draft", draft_config)
            return validate_mtp_eagle3_checkpoint_pair(
                target,
                draft,
                algorithm,
                num_speculative_tokens=1 if algorithm == "MTP" else 3,
            )

    def test_valid_mtp_pair_passes(self) -> None:
        result = self._validate(_mtp_target_config(), _mtp_draft_config(), "MTP")

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["target_mtp_layer_count"], 1)
        self.assertEqual(len(result["checks"]), 25)
        self.assertTrue(result["weight_layout"]["target_inspected"])
        self.assertTrue(result["weight_layout"]["draft_inspected"])
        self.assertEqual(
            result["weight_layout"]["target_hc_head_fn_shape"],
            [4, 16384],
        )
        self.assertEqual(result["weight_layout"]["draft_e_proj_shape"], [4096, 4096])

    def test_mtp_target_and_draft_hc_mult_must_match(self) -> None:
        draft = _mtp_draft_config()
        draft["hc_mult"] = 3

        result = self._validate(_mtp_target_config(), draft, "MTP")

        self.assertFalse(result["passed"])
        self.assertEqual(result["target_hc_mult"], 4)
        self.assertEqual(result["draft_hc_mult"], 3)
        self.assertIn("hc_mult_matches", result["failures"])
        self.assertTrue(result["checks"]["draft_hc_head_shapes_match"])

    def test_mtp_missing_indexed_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _mtp_target_config())
            draft = _write_checkpoint(root, "draft", _mtp_draft_config())
            _write_fake_safetensors_index(
                target / "model.safetensors.index.json",
                {"unused.weight": "missing.safetensors"},
            )

            result = validate_mtp_eagle3_checkpoint_pair(target, draft, "MTP", num_speculative_tokens=1)

        self.assertFalse(result["passed"])
        self.assertIn("target_safetensors_shards_exist", result["failures"])
        self.assertEqual(result["weight_layout"]["target_safetensors_shard_count"], 2)

    def test_mtp_target_weight_layout_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _mtp_target_config())
            draft = _write_checkpoint(root, "draft", _mtp_draft_config())
            _write_fake_safetensors(
                target / "model.safetensors",
                {
                    "model.embed.weight": [1, 1],
                    "model.layers.0.attn_norm.weight": [1],
                    "model.norm.weight": [1],
                    "model.head.weight": [1, 1],
                },
            )

            result = validate_mtp_eagle3_checkpoint_pair(target, draft, "MTP", num_speculative_tokens=1)

        self.assertFalse(result["passed"])
        self.assertIn("target_weight_prefixes_match_runtime", result["failures"])
        self.assertIn("target_embedding_shape_matches", result["failures"])
        self.assertIn("target_final_norm_shape_matches", result["failures"])
        self.assertIn("target_lm_head_shape_matches", result["failures"])
        self.assertIn("target_hc_head_shapes_match", result["failures"])

    def test_mtp_draft_weight_layout_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _mtp_target_config())
            draft = _write_checkpoint(root, "draft", _mtp_draft_config())
            _write_fake_safetensors(
                draft / "model.safetensors",
                {
                    "layers.0.enorm.weight": [1],
                    "layers.0.e_proj.weight": [1, 1],
                    "layers.0.hc_head_fn": [1, 1],
                },
            )

            result = validate_mtp_eagle3_checkpoint_pair(target, draft, "MTP", num_speculative_tokens=1)

        self.assertFalse(result["passed"])
        self.assertIn("draft_weight_prefixes_match_runtime", result["failures"])
        self.assertIn("draft_norm_shapes_match", result["failures"])
        self.assertIn("draft_projection_shapes_match", result["failures"])
        self.assertIn("draft_hc_head_shapes_match", result["failures"])

    def test_nested_qwen_text_target_family_is_supported(self) -> None:
        target_model = _mtp_target_config()
        target_model["model_type"] = "qwen3_5_text"
        target_model["mtp_num_hidden_layers"] = target_model.pop("num_nextn_predict_layers")
        target = {"model_type": "qwen3_5", "text_config": target_model}
        draft_model = _mtp_draft_config()
        draft_model["model_type"] = "qwen3_5_text"
        draft_model["mtp_num_hidden_layers"] = draft_model.pop("num_nextn_predict_layers")
        draft = {
            "architectures": ["Qwen3_5MtpForCausalLM"],
            "model_type": "qwen3_5_mtp",
            "text_config": draft_model,
            "bos_token_id": 0,
            "eos_token_id": 1,
        }

        result = self._validate(target, draft, "MTP")

        self.assertTrue(result["passed"])

    def test_mtp_wrong_runtime_type_and_family_are_rejected(self) -> None:
        draft = _mtp_draft_config()
        draft["model_type"] = "unregistered_mtp"

        result = self._validate(_mtp_target_config(), draft, "MTP")

        self.assertFalse(result["passed"])
        self.assertIn("draft_runtime_model_type_supported", result["failures"])
        self.assertIn("model_family_matches", result["failures"])

    def test_mtp_hidden_vocab_and_layer_mismatches_are_rejected(self) -> None:
        draft = _mtp_draft_config()
        draft["hidden_size"] = 2048
        draft["vocab_size"] = 32000
        draft["num_hidden_layers"] = 2
        draft["num_nextn_predict_layers"] = 2

        result = self._validate(_mtp_target_config(), draft, "MTP")

        self.assertFalse(result["passed"])
        self.assertIn("hidden_size_matches", result["failures"])
        self.assertIn("vocab_size_matches", result["failures"])
        self.assertIn("draft_layer_count_matches_target_mtp_layers", result["failures"])
        self.assertIn("draft_declared_mtp_layers_match", result["failures"])

    def test_mtp_target_must_declare_predict_layers(self) -> None:
        target = _mtp_target_config()
        target.pop("num_nextn_predict_layers")

        result = self._validate(target, _mtp_draft_config(), "MTP")

        self.assertFalse(result["passed"])
        self.assertIn("target_declares_mtp_layers", result["failures"])

    def test_valid_kimi_eagle3_pair_passes(self) -> None:
        result = self._validate(_eagle3_target_config(), _eagle3_draft_config(), "Eagle3")

        self.assertTrue(result["passed"])
        self.assertEqual(result["target_capture_layer_ids"], [2, 30, 58])
        self.assertEqual(len(result["checks"]), 24)
        self.assertEqual(result["weight_layout"]["draft_fc_shape"], [7168, 21504])
        self.assertIsNone(result["weight_layout"]["draft_d2t_shape"])

    def test_eagle3_vocab_subset_requires_complete_d2t_mapping(self) -> None:
        draft_config = _eagle3_draft_config()
        draft_config["draft_vocab_size"] = 160000

        missing_mapping = self._validate(_eagle3_target_config(), draft_config, "Eagle3")
        self.assertFalse(missing_mapping["passed"])
        self.assertIn(
            "draft_token_mapping_matches_vocab",
            missing_mapping["failures"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _eagle3_target_config())
            draft = _write_checkpoint(root, "draft", draft_config)
            draft_shapes = _fake_weight_shapes(draft_config)
            assert draft_shapes is not None
            draft_shapes["d2t"] = [160000]
            _write_fake_safetensors(draft / "model.safetensors", draft_shapes)

            mapped_subset = validate_mtp_eagle3_checkpoint_pair(
                target,
                draft,
                "Eagle3",
                num_speculative_tokens=3,
            )

        self.assertTrue(mapped_subset["passed"])
        self.assertEqual(
            mapped_subset["weight_layout"]["draft_d2t_shape"],
            [160000],
        )

    def test_kimi_target_weight_layout_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _eagle3_target_config())
            draft = _write_checkpoint(root, "draft", _eagle3_draft_config())
            _write_fake_safetensors(
                target / "model.safetensors",
                {
                    "model.embed_tokens.weight": [1, 1],
                    "model.layers.0.input_layernorm.weight": [1],
                    "model.norm.weight": [1],
                    "lm_head.weight": [1, 1],
                },
            )

            result = validate_mtp_eagle3_checkpoint_pair(target, draft, "Eagle3", num_speculative_tokens=3)

        self.assertFalse(result["passed"])
        self.assertIn("target_weight_prefixes_match_runtime", result["failures"])
        self.assertIn("target_embedding_shape_matches", result["failures"])
        self.assertIn("target_final_norm_shape_matches", result["failures"])
        self.assertIn("target_lm_head_shape_matches", result["failures"])

    def test_kimi_draft_weight_layout_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _eagle3_target_config())
            draft = _write_checkpoint(root, "draft", _eagle3_draft_config())
            _write_fake_safetensors(
                draft / "model.safetensors",
                {
                    "fc.weight": [1, 1],
                    "norm.weight": [1],
                    "lm_head.weight": [1, 1],
                },
            )

            result = validate_mtp_eagle3_checkpoint_pair(target, draft, "Eagle3", num_speculative_tokens=3)

        self.assertFalse(result["passed"])
        self.assertIn("draft_weight_modules_match_runtime", result["failures"])
        self.assertIn("draft_fc_shape_matches_3h", result["failures"])
        self.assertIn("draft_final_norm_shape_matches", result["failures"])
        self.assertIn("draft_lm_head_shape_matches", result["failures"])

    def test_qwen3_eagle3_runtime_type_is_supported(self) -> None:
        draft = _eagle3_draft_config()
        draft["model_type"] = "qwen3_eagle3"

        result = self._validate(_eagle3_target_config(), draft, "Eagle3")

        self.assertTrue(result["passed"])

    def test_eagle3_wrong_architecture_and_runtime_type_are_rejected(self) -> None:
        target = _eagle3_target_config()
        target["model_type"] = "unsupported_vlm"
        draft = _eagle3_draft_config()
        draft["architectures"] = ["LlamaForCausalLM"]
        draft["model_type"] = "llama"

        result = self._validate(target, draft, "Eagle3")

        self.assertFalse(result["passed"])
        self.assertIn("target_runtime_model_type_supported", result["failures"])
        self.assertIn("draft_architecture_matches_algorithm", result["failures"])
        self.assertIn("draft_runtime_model_type_supported", result["failures"])

    def test_eagle3_requires_one_layer_and_compatible_vocab(self) -> None:
        draft = _eagle3_draft_config()
        draft["num_hidden_layers"] = 2
        draft["draft_vocab_size"] = 163841

        result = self._validate(_eagle3_target_config(), draft, "Eagle3")

        self.assertFalse(result["passed"])
        self.assertIn("draft_layer_count_is_one", result["failures"])
        self.assertIn("draft_vocab_is_target_subset", result["failures"])

    def test_eagle3_target_must_supply_three_unique_capture_layers(self) -> None:
        target = _eagle3_target_config()
        target["text_config"]["num_hidden_layers"] = 4

        result = self._validate(target, _eagle3_draft_config(), "Eagle3")

        self.assertFalse(result["passed"])
        self.assertIn("target_capture_layers_are_unique", result["failures"])

    def test_special_token_mismatches_are_rejected(self) -> None:
        draft = _eagle3_draft_config()
        draft["bos_token_id"] = 1
        draft["eos_token_id"] = 2

        result = self._validate(_eagle3_target_config(), draft, "Eagle3")

        self.assertFalse(result["passed"])
        self.assertIn("bos_token_id_matches", result["failures"])
        self.assertIn("eos_token_id_matches", result["failures"])

    def test_missing_draft_weights_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _mtp_target_config())
            draft = _write_checkpoint(root, "draft", _mtp_draft_config(), with_weights=False)

            result = validate_mtp_eagle3_checkpoint_pair(target, draft, "MTP", num_speculative_tokens=1)

        self.assertFalse(result["passed"])
        self.assertIn("draft_weights_present", result["failures"])

    def test_target_and_draft_paths_must_be_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = _write_checkpoint(root, "model", _mtp_draft_config())

            result = validate_mtp_eagle3_checkpoint_pair(checkpoint, checkpoint, "MTP", num_speculative_tokens=1)

        self.assertFalse(result["passed"])
        self.assertIn("checkpoint_paths_are_distinct", result["failures"])


if __name__ == "__main__":
    unittest.main()
