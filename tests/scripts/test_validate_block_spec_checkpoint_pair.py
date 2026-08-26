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

from tools.validate_block_spec_checkpoint_pair import (
    validate_block_spec_checkpoint_pair,
)


def _target_config() -> dict[str, Any]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "head_dim": 128,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "vocab_size": 151936,
    }


def _draft_config(algorithm: str = "DFlash") -> dict[str, Any]:
    return {
        "architectures": [f"{algorithm}DraftModel"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "head_dim": 128,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_hidden_layers": 5,
        "num_key_value_heads": 8,
        "vocab_size": 151936,
        "num_target_layers": 36,
        "block_size": 16,
        "dflash_config": {
            "mask_token_id": 151669,
            "target_layer_ids": [1, 9, 17, 25, 33],
        },
    }


def _qwen35_target_config() -> dict[str, Any]:
    target_model = _target_config()
    target_model.update(
        {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "head_dim": 256,
            "intermediate_size": 17408,
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "vocab_size": 248320,
        }
    )
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": target_model,
    }


def _qwen35_draft_config() -> dict[str, Any]:
    draft = _draft_config()
    draft.update(
        {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_target_layers": 64,
        }
    )
    draft["dflash_config"] = {
        "mask_token_id": 248070,
        "target_layer_ids": [1, 16, 31, 46, 61],
    }
    return draft


def _dspark_draft_config() -> dict[str, Any]:
    draft = _draft_config()
    draft.update(
        {
            "architectures": ["Qwen3DSparkModel"],
            "block_size": 7,
            "markov_rank": 256,
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "target_layer_ids": [1, 9, 17, 25, 33],
            "mask_token_id": 151669,
        }
    )
    draft.pop("dflash_config")
    return draft


def _deepseek_dspark_target_config() -> dict[str, Any]:
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "vocab_size": 129280,
        "hc_mult": 4,
    }


def _deepseek_dspark_draft_config() -> dict[str, Any]:
    draft = _deepseek_dspark_target_config()
    draft.update(
        {
            "dspark_markov_rank": 256,
            "dspark_target_layer_ids": [40, 41, 42],
            "dspark_noise_token_id": 128799,
            "dspark_block_size": 5,
        }
    )
    return draft


def _target_weight_shapes(
    config: dict[str, Any],
) -> dict[str, list[int]]:
    model = config.get("text_config", config)
    model_type = config.get("model_type", model.get("model_type"))
    prefix = "model.language_model." if model_type in {"qwen3_5", "qwen3_5_text"} else "model."
    hidden_size = model["hidden_size"]
    vocab_size = model["vocab_size"]
    last_layer = model["num_hidden_layers"] - 1
    shapes = {
        prefix + "embed_tokens.weight": [vocab_size, hidden_size],
        prefix + "norm.weight": [hidden_size],
        prefix + "layers.0.input_layernorm.weight": [hidden_size],
        prefix + f"layers.{last_layer}.input_layernorm.weight": [hidden_size],
    }
    if not model.get("tie_word_embeddings", config.get("tie_word_embeddings", False)):
        shapes["lm_head.weight"] = [vocab_size, hidden_size]
    return shapes


def _draft_weight_shapes(config: dict[str, Any], prefix: str = "") -> dict[str, list[int]]:
    hidden_size = config["hidden_size"]
    dflash_config = config.get("dflash_config", {})
    capture_layer_ids = dflash_config.get("target_layer_ids", config.get("target_layer_ids", []))
    capture_count = (
        len(capture_layer_ids)
        if isinstance(capture_layer_ids, list) and capture_layer_ids
        else config["num_hidden_layers"]
    )
    head_dim = config["head_dim"]
    query_width = config["num_attention_heads"] * head_dim
    kv_width = config["num_key_value_heads"] * head_dim
    intermediate_size = config["intermediate_size"]
    shapes = {
        prefix + "fc.weight": [hidden_size, hidden_size * capture_count],
        prefix + "hidden_norm.weight": [hidden_size],
        prefix + "norm.weight": [hidden_size],
    }
    for layer_index in range(config["num_hidden_layers"]):
        layer_prefix = prefix + f"layers.{layer_index}."
        shapes.update(
            {
                layer_prefix + "self_attn.q_proj.weight": [
                    query_width,
                    hidden_size,
                ],
                layer_prefix + "self_attn.k_proj.weight": [
                    kv_width,
                    hidden_size,
                ],
                layer_prefix + "self_attn.v_proj.weight": [
                    kv_width,
                    hidden_size,
                ],
                layer_prefix + "self_attn.o_proj.weight": [
                    hidden_size,
                    query_width,
                ],
                layer_prefix + "self_attn.q_norm.weight": [head_dim],
                layer_prefix + "self_attn.k_norm.weight": [head_dim],
                layer_prefix + "mlp.gate_proj.weight": [
                    intermediate_size,
                    hidden_size,
                ],
                layer_prefix + "mlp.up_proj.weight": [
                    intermediate_size,
                    hidden_size,
                ],
                layer_prefix + "mlp.down_proj.weight": [
                    hidden_size,
                    intermediate_size,
                ],
                layer_prefix + "input_layernorm.weight": [hidden_size],
                layer_prefix + "post_attention_layernorm.weight": [hidden_size],
            }
        )
    markov_rank = config.get("markov_rank")
    if isinstance(markov_rank, int) and markov_rank > 0:
        shapes.update(
            {
                prefix + "markov_head.markov_w1.weight": [
                    config["vocab_size"],
                    markov_rank,
                ],
                prefix + "markov_head.markov_w2.weight": [
                    config["vocab_size"],
                    markov_rank,
                ],
            }
        )
        if config.get("enable_confidence_head", False):
            confidence_input_width = hidden_size
            if config.get("confidence_head_with_markov", False):
                confidence_input_width += markov_rank
            shapes.update(
                {
                    prefix + "confidence_head.proj.weight": [
                        1,
                        confidence_input_width,
                    ],
                    prefix + "confidence_head.proj.bias": [1],
                }
            )
    return shapes


def _deepseek_target_weight_shapes(
    config: dict[str, Any],
) -> dict[str, list[int]]:
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


def _deepseek_dspark_weight_shapes(
    config: dict[str, Any],
) -> dict[str, list[int]]:
    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]
    layer_count = len(config["dspark_target_layer_ids"])
    last_layer = layer_count - 1
    hc_mult = config.get("hc_mult", 4)
    markov_rank = config["dspark_markov_rank"]
    shapes: dict[str, list[int]] = {}
    for layer_index in range(layer_count):
        prefix = f"mtp.{layer_index}."
        shapes.update(
            {
                prefix + "attn_norm.weight": [hidden_size],
                prefix + "ffn_norm.weight": [hidden_size],
                prefix + "attn.wq_a.weight": [1024, hidden_size],
                prefix + "ffn.gate.weight": [256, hidden_size],
            }
        )
    shapes.update(
        {
            "mtp.0.main_proj.weight": [
                hidden_size,
                hidden_size * layer_count,
            ],
            "mtp.0.main_norm.weight": [hidden_size],
            "mtp.0.embed.weight": [vocab_size, hidden_size],
            f"mtp.{last_layer}.norm.weight": [hidden_size],
            f"mtp.{last_layer}.head.weight": [vocab_size, hidden_size],
            f"mtp.{last_layer}.hc_head_fn": [
                hc_mult,
                hc_mult * hidden_size,
            ],
            f"mtp.{last_layer}.hc_head_base": [hc_mult],
            f"mtp.{last_layer}.hc_head_scale": [1],
            f"mtp.{last_layer}.markov_head.markov_w1.weight": [
                vocab_size,
                markov_rank,
            ],
            f"mtp.{last_layer}.markov_head.markov_w2.weight": [
                vocab_size,
                markov_rank,
            ],
            f"mtp.{last_layer}.confidence_head.proj.weight": [
                1,
                hidden_size + markov_rank,
            ],
        }
    )
    return shapes


def _fake_weight_shapes(config: dict[str, Any], checkpoint_name: str) -> dict[str, list[int]]:
    if config.get("model_type") == "deepseek_v4":
        if checkpoint_name == "draft" and config.get("dspark_target_layer_ids"):
            return _deepseek_dspark_weight_shapes(config)
        return _deepseek_target_weight_shapes(config)
    architectures = config.get("architectures", [])
    is_block_spec_draft = any(
        isinstance(architecture, str) and ("dflash" in architecture.lower() or "dspark" in architecture.lower())
        for architecture in architectures
    )
    if checkpoint_name == "draft" and is_block_spec_draft:
        return _draft_weight_shapes(config)
    return _target_weight_shapes(config)


def _write_fake_safetensors(path: Path, weight_shapes: dict[str, list[int]]) -> None:
    header = {
        key: {
            "dtype": "BF16",
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
        _write_fake_safetensors(
            directory / "model.safetensors",
            _fake_weight_shapes(config, name),
        )
    return directory


class ValidateBlockSpecCheckpointPairTest(unittest.TestCase):
    def _validate(
        self,
        target_config: dict[str, Any],
        draft_config: dict[str, Any],
        algorithm: str = "DFlash",
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", target_config)
            draft = _write_checkpoint(root, "draft", draft_config)
            return validate_block_spec_checkpoint_pair(
                target,
                draft,
                algorithm,
                num_speculative_tokens=4,
            )

    def test_valid_dflash_pair_passes(self) -> None:
        result = self._validate(_target_config(), _draft_config())

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["checkpoint_block_size"], 16)
        self.assertEqual(len(result["checks"]), 28)
        self.assertTrue(result["weight_layout"]["target_inspected"])
        self.assertTrue(result["weight_layout"]["draft_inspected"])
        self.assertEqual(result["weight_layout"]["draft_fc_shape"], [4096, 20480])
        self.assertEqual(result["weight_layout"]["draft_layer_count_inspected"], 5)
        self.assertTrue(result["weight_layout"]["draft_uses_target_shared_embedding"])
        self.assertTrue(result["weight_layout"]["draft_uses_target_shared_lm_head"])

    def test_valid_qwen35_dflash_pair_passes(self) -> None:
        result = self._validate(_qwen35_target_config(), _qwen35_draft_config())

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["weight_layout"]["target_model_prefix"],
            "model.language_model.",
        )
        self.assertEqual(result["weight_layout"]["draft_fc_shape"], [5120, 25600])

    def test_valid_qwen3_dspark_pair_passes(self) -> None:
        result = self._validate(_target_config(), _dspark_draft_config(), algorithm="DSpark")

        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 32)
        self.assertEqual(result["weight_layout"]["draft_markov_rank"], 256)
        self.assertEqual(
            result["weight_layout"]["draft_markov_w1_shape"],
            [151936, 256],
        )
        self.assertEqual(
            result["weight_layout"]["draft_confidence_weight_shape"],
            [1, 4352],
        )

    def test_tied_target_embedding_supplies_lm_head(self) -> None:
        target = _target_config()
        target["tie_word_embeddings"] = True

        result = self._validate(target, _dspark_draft_config(), algorithm="DSpark")

        self.assertTrue(result["passed"])
        self.assertEqual(result["weight_layout"]["target_lm_head_source"], "embedding")
        self.assertEqual(
            result["weight_layout"]["target_lm_head_shape"],
            [151936, 4096],
        )

    def test_dspark_markov_and_confidence_shapes_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _target_config())
            draft_config = _dspark_draft_config()
            draft = _write_checkpoint(root, "draft", draft_config)
            shapes = _draft_weight_shapes(draft_config)
            shapes["markov_head.markov_w1.weight"] = [1, 1]
            shapes["confidence_head.proj.weight"] = [1, 4096]
            _write_fake_safetensors(draft / "model.safetensors", shapes)

            result = validate_block_spec_checkpoint_pair(target, draft, "DSpark", num_speculative_tokens=4)

        self.assertFalse(result["passed"])
        self.assertIn("draft_markov_shapes_match", result["failures"])
        self.assertIn("draft_confidence_shapes_match", result["failures"])

    def test_dspark_confidence_bias_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _target_config())
            draft_config = _dspark_draft_config()
            draft = _write_checkpoint(root, "draft", draft_config)
            shapes = _draft_weight_shapes(draft_config)
            shapes.pop("confidence_head.proj.bias")
            _write_fake_safetensors(draft / "model.safetensors", shapes)

            result = validate_block_spec_checkpoint_pair(target, draft, "DSpark", num_speculative_tokens=4)

        self.assertTrue(result["passed"])

    def test_model_prefixed_draft_weights_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _target_config())
            draft_config = _draft_config()
            draft = _write_checkpoint(root, "draft", draft_config)
            _write_fake_safetensors(
                draft / "model.safetensors",
                _draft_weight_shapes(draft_config, prefix="model."),
            )

            result = validate_block_spec_checkpoint_pair(target, draft, "DFlash", num_speculative_tokens=4)

        self.assertTrue(result["passed"])
        self.assertEqual(result["weight_layout"]["draft_model_prefix"], "model.")

    def test_target_weight_layout_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _target_config())
            draft = _write_checkpoint(root, "draft", _draft_config())
            _write_fake_safetensors(
                target / "model.safetensors",
                {
                    "embed_tokens.weight": [1, 1],
                    "norm.weight": [1],
                    "layers.0.input_layernorm.weight": [1],
                    "lm_head.weight": [1, 1],
                },
            )

            result = validate_block_spec_checkpoint_pair(target, draft, "DFlash", num_speculative_tokens=4)

        self.assertFalse(result["passed"])
        self.assertIn("target_weight_prefixes_match_runtime", result["failures"])
        self.assertIn("target_embedding_shape_matches", result["failures"])
        self.assertIn("target_final_norm_shape_matches", result["failures"])
        self.assertIn("target_lm_head_shape_matches", result["failures"])
        self.assertIn("target_boundary_layer_norm_shapes_match", result["failures"])

    def test_draft_weight_layout_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _target_config())
            draft = _write_checkpoint(root, "draft", _draft_config())
            _write_fake_safetensors(
                draft / "model.safetensors",
                {
                    "wrong.fc.weight": [1, 1],
                    "wrong.hidden_norm.weight": [1],
                    "wrong.layers.0.self_attn.q_proj.weight": [1, 1],
                    "wrong.layers.0.mlp.gate_proj.weight": [1, 1],
                    "wrong.norm.weight": [1],
                },
            )

            result = validate_block_spec_checkpoint_pair(target, draft, "DFlash", num_speculative_tokens=4)

        self.assertFalse(result["passed"])
        self.assertIn("draft_weight_prefix_matches_runtime", result["failures"])
        self.assertIn("draft_required_modules_match_runtime", result["failures"])
        self.assertIn("draft_fc_shape_matches_capture_width", result["failures"])
        self.assertIn("draft_norm_shapes_match", result["failures"])
        self.assertIn("draft_attention_shapes_match", result["failures"])
        self.assertIn("draft_mlp_shapes_match", result["failures"])
        self.assertIn("draft_layer_norm_shapes_match", result["failures"])

    def test_missing_indexed_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _target_config())
            draft = _write_checkpoint(root, "draft", _draft_config())
            _write_fake_safetensors_index(
                draft / "model.safetensors.index.json",
                {"unused.weight": "missing.safetensors"},
            )

            result = validate_block_spec_checkpoint_pair(target, draft, "DFlash", num_speculative_tokens=4)

        self.assertFalse(result["passed"])
        self.assertIn("draft_safetensors_shards_exist", result["failures"])
        self.assertEqual(result["weight_layout"]["draft_safetensors_shard_count"], 2)

    def test_hidden_and_vocab_mismatches_are_rejected(self) -> None:
        draft = _draft_config()
        draft["hidden_size"] = 2048
        draft["vocab_size"] = 32000

        result = self._validate(_target_config(), draft)

        self.assertFalse(result["passed"])
        self.assertIn("hidden_size_matches", result["failures"])
        self.assertIn("vocab_size_matches", result["failures"])

    def test_invalid_capture_layers_are_rejected(self) -> None:
        draft = _draft_config()
        draft["dflash_config"]["target_layer_ids"] = [1, 1, 36]

        result = self._validate(_target_config(), draft)

        self.assertFalse(result["passed"])
        self.assertIn("capture_layers_are_unique", result["failures"])
        self.assertIn("capture_layers_are_in_range", result["failures"])

    def test_draft_layer_count_must_match_capture_count(self) -> None:
        draft = _draft_config()
        draft["num_hidden_layers"] = 4

        result = self._validate(_target_config(), draft)

        self.assertFalse(result["passed"])
        self.assertIn("draft_layer_count_matches_capture_layers", result["failures"])

    def test_invalid_mask_token_is_rejected(self) -> None:
        draft = _draft_config()
        draft["dflash_config"]["mask_token_id"] = draft["vocab_size"]

        result = self._validate(_target_config(), draft)

        self.assertFalse(result["passed"])
        self.assertIn("mask_token_is_valid", result["failures"])

    def test_wrong_algorithm_architecture_is_rejected(self) -> None:
        result = self._validate(_target_config(), _draft_config(), algorithm="DSpark")

        self.assertFalse(result["passed"])
        self.assertIn("draft_architecture_matches_algorithm", result["failures"])

    def test_deepseek_v4_dspark_aliases_pass(self) -> None:
        result = self._validate(
            _deepseek_dspark_target_config(),
            _deepseek_dspark_draft_config(),
            algorithm="DSpark",
        )

        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 32)
        self.assertEqual(result["draft_layer_count"], 43)
        self.assertEqual(result["draft_effective_layer_count"], 3)
        self.assertEqual(
            result["weight_layout"]["draft_main_proj_shape"],
            [4096, 12288],
        )
        self.assertEqual(result["weight_layout"]["draft_embedding_source"], "dedicated")
        self.assertEqual(result["weight_layout"]["draft_lm_head_source"], "dedicated")

    def test_deepseek_v4_dspark_layout_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _deepseek_dspark_target_config())
            draft_config = _deepseek_dspark_draft_config()
            draft = _write_checkpoint(root, "draft", draft_config)
            shapes = _deepseek_dspark_weight_shapes(draft_config)
            shapes["mtp.0.main_proj.weight"] = [1, 1]
            shapes["mtp.2.markov_head.markov_w1.weight"] = [1, 1]
            shapes["mtp.2.confidence_head.proj.weight"] = [1, 4096]
            shapes.pop("mtp.1.ffn.gate.weight")
            _write_fake_safetensors(draft / "model.safetensors", shapes)

            result = validate_block_spec_checkpoint_pair(target, draft, "DSpark", num_speculative_tokens=4)

        self.assertFalse(result["passed"])
        self.assertIn("draft_weight_prefixes_match_runtime", result["failures"])
        self.assertIn("draft_main_projection_shapes_match", result["failures"])
        self.assertIn("draft_markov_shapes_match", result["failures"])
        self.assertIn("draft_confidence_shapes_match", result["failures"])

    def test_nested_target_text_config_is_supported(self) -> None:
        target = {"text_config": _target_config()}

        result = self._validate(target, _draft_config())

        self.assertTrue(result["passed"])

    def test_missing_draft_weights_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = _write_checkpoint(root, "target", _target_config())
            draft = _write_checkpoint(root, "draft", _draft_config(), with_weights=False)

            result = validate_block_spec_checkpoint_pair(target, draft, "DFlash", num_speculative_tokens=4)

        self.assertFalse(result["passed"])
        self.assertIn("draft_weights_present", result["failures"])

    def test_declared_layers_and_block_size_are_validated(self) -> None:
        draft = _draft_config()
        draft["num_target_layers"] = 35
        draft["block_size"] = 0

        result = self._validate(_target_config(), draft)

        self.assertFalse(result["passed"])
        self.assertIn("declared_target_layers_match", result["failures"])
        self.assertIn("checkpoint_block_size_is_valid", result["failures"])

    def test_missing_capture_layers_and_mask_token_are_rejected(self) -> None:
        draft = _draft_config()
        draft.pop("dflash_config")

        result = self._validate(_target_config(), draft)

        self.assertFalse(result["passed"])
        self.assertIn("capture_layers_are_integers", result["failures"])
        self.assertIn("mask_token_is_valid", result["failures"])

    def test_target_and_draft_paths_must_be_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = _write_checkpoint(root, "model", _draft_config())

            result = validate_block_spec_checkpoint_pair(
                checkpoint,
                checkpoint,
                "DFlash",
                num_speculative_tokens=4,
            )

        self.assertFalse(result["passed"])
        self.assertIn("checkpoint_paths_are_distinct", result["failures"])


if __name__ == "__main__":
    unittest.main()
