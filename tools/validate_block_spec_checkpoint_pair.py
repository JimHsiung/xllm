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
"""Validate a DFlash/DSpark target and draft checkpoint pair before startup."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger
from tools.safetensors_manifest import SafetensorsManifest

_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.index.json",
)

_QWEN_TARGET_PREFIXES = {
    "qwen3": "model.",
    "qwen3_5": "model.language_model.",
    "qwen3_5_text": "model.language_model.",
}


def _load_config(model_dir: Path, label: str) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise ValueError(f"{label} checkpoint directory does not exist: {model_dir}")
    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"{label} config root must be a JSON object: {config_path}")
    return config


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def _required_positive_int(config: dict[str, Any], key: str, context: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer")
    return value


def _first_value(config: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        value: Any = config
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def _has_weights(model_dir: Path) -> bool:
    return any(path.is_file() for pattern in _WEIGHT_PATTERNS for path in model_dir.glob(pattern))


def _draft_architecture_matches(config: dict[str, Any], algorithm: str) -> bool:
    architectures = config.get("architectures")
    architecture_names = architectures if isinstance(architectures, list) else []
    expected = f"{algorithm}DraftModel"
    if expected in architecture_names:
        return True
    if any(
        isinstance(architecture, str) and algorithm.lower() in architecture.lower()
        for architecture in architecture_names
    ):
        return True
    model_type = config.get("model_type")
    return algorithm == "DSpark" and isinstance(model_type, str) and "deepseek_v4" in model_type.lower()


def _qwen_target_model_type(target: dict[str, Any], target_model: dict[str, Any]) -> str | None:
    for config in (target, target_model):
        model_type = config.get("model_type")
        if model_type in _QWEN_TARGET_PREFIXES:
            return model_type
    return None


def _qwen_draft_prefix(manifest: SafetensorsManifest) -> str | None:
    if manifest.has_weight("model.fc.weight"):
        return "model."
    if manifest.has_weight("fc.weight"):
        return ""
    return None


def _required_bool(config: dict[str, Any], key: str, default: bool, context: str) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a boolean")
    return value


def _positive_int_or_default(config: dict[str, Any], key: str, default: int, context: str) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer")
    return value


def _is_deepseek_v4_dspark(algorithm: str, draft_model: dict[str, Any]) -> bool:
    model_type = draft_model.get("model_type")
    return algorithm == "DSpark" and isinstance(model_type, str) and "deepseek_v4" in model_type.lower()


def _deepseek_v4_dspark_weight_contract(
    target_dir: Path,
    draft_dir: Path,
    target_model: dict[str, Any],
    draft_model: dict[str, Any],
    target_hidden_size: int,
    draft_hidden_size: int,
    target_vocab_size: int,
    draft_vocab_size: int,
    target_layer_count: int,
    draft_layer_count: int,
) -> tuple[dict[str, bool], dict[str, Any]]:
    target_manifest = SafetensorsManifest(target_dir, "target")
    target_embedding_shape = target_manifest.shape("embed.weight")
    target_final_norm_shape = target_manifest.shape("norm.weight")
    target_lm_head_shape = target_manifest.shape("head.weight")
    target_hc_head_fn_shape = target_manifest.shape("hc_head_fn")
    target_hc_head_base_shape = target_manifest.shape("hc_head_base")
    target_hc_head_scale_shape = target_manifest.shape("hc_head_scale")
    target_hc_mult = _positive_int_or_default(target_model, "hc_mult", 4, "target config")

    draft_manifest = SafetensorsManifest(draft_dir, "draft")
    draft_hc_mult = _positive_int_or_default(draft_model, "hc_mult", 4, "draft config")
    draft_markov_rank = _required_positive_int(draft_model, "dspark_markov_rank", "draft config")
    first_prefix = "mtp.0."
    last_prefix = f"mtp.{draft_layer_count - 1}."
    draft_main_proj_shape = draft_manifest.shape(first_prefix + "main_proj.weight")
    draft_main_norm_shape = draft_manifest.shape(first_prefix + "main_norm.weight")

    dedicated_embedding_key = first_prefix + "embed.weight"
    draft_embedding_key = (
        dedicated_embedding_key if draft_manifest.has_weight(dedicated_embedding_key) else "embed.weight"
    )
    dedicated_lm_head_key = last_prefix + "head.weight"
    draft_lm_head_key = dedicated_lm_head_key if draft_manifest.has_weight(dedicated_lm_head_key) else "head.weight"
    draft_embedding_shape = draft_manifest.shape(draft_embedding_key)
    draft_final_norm_shape = draft_manifest.shape(last_prefix + "norm.weight")
    draft_lm_head_shape = draft_manifest.shape(draft_lm_head_key)
    draft_hc_head_fn_shape = draft_manifest.shape(last_prefix + "hc_head_fn")
    draft_hc_head_base_shape = draft_manifest.shape(last_prefix + "hc_head_base")
    draft_hc_head_scale_shape = draft_manifest.shape(last_prefix + "hc_head_scale")
    draft_markov_w1_shape = draft_manifest.shape(last_prefix + "markov_head.markov_w1.weight")
    draft_markov_w2_shape = draft_manifest.shape(last_prefix + "markov_head.markov_w2.weight")
    draft_confidence_weight_shape = draft_manifest.shape(last_prefix + "confidence_head.proj.weight")
    draft_confidence_bias_shape = draft_manifest.shape(last_prefix + "confidence_head.proj.bias")

    draft_layer_modules_match_runtime = True
    draft_layer_norm_shapes_match = True
    draft_layer_norm_shapes: list[dict[str, Any]] = []
    for layer_index in range(draft_layer_count):
        layer_prefix = f"mtp.{layer_index}."
        attn_norm_key = layer_prefix + "attn_norm.weight"
        ffn_norm_key = layer_prefix + "ffn_norm.weight"
        attn_norm_shape = draft_manifest.shape(attn_norm_key)
        ffn_norm_shape = draft_manifest.shape(ffn_norm_key)
        draft_layer_modules_match_runtime = (
            draft_layer_modules_match_runtime
            and draft_manifest.has_weight(attn_norm_key)
            and draft_manifest.has_weight(ffn_norm_key)
            and draft_manifest.has_prefix(layer_prefix + "attn.")
            and draft_manifest.has_prefix(layer_prefix + "ffn.")
        )
        draft_layer_norm_shapes_match = (
            draft_layer_norm_shapes_match
            and attn_norm_shape == [draft_hidden_size]
            and ffn_norm_shape == [draft_hidden_size]
        )
        draft_layer_norm_shapes.append(
            {
                "layer_index": layer_index,
                "attn_norm": attn_norm_shape,
                "ffn_norm": ffn_norm_shape,
            }
        )

    target_required_keys = (
        "embed.weight",
        "norm.weight",
        "head.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    )
    draft_required_keys = (
        first_prefix + "main_proj.weight",
        first_prefix + "main_norm.weight",
        draft_embedding_key,
        last_prefix + "norm.weight",
        draft_lm_head_key,
        last_prefix + "hc_head_fn",
        last_prefix + "hc_head_base",
        last_prefix + "hc_head_scale",
    )
    draft_markov_keys = (
        last_prefix + "markov_head.markov_w1.weight",
        last_prefix + "markov_head.markov_w2.weight",
    )
    draft_confidence_weight_key = last_prefix + "confidence_head.proj.weight"
    checks = {
        "target_weight_prefixes_match_runtime": all(target_manifest.has_weight(key) for key in target_required_keys)
        and target_manifest.has_prefix("layers.0.")
        and target_manifest.has_prefix(f"layers.{target_layer_count - 1}."),
        "target_embedding_shape_matches": target_embedding_shape == [target_vocab_size, target_hidden_size],
        "target_final_norm_shape_matches": target_final_norm_shape == [target_hidden_size],
        "target_lm_head_shape_matches": target_lm_head_shape == [target_vocab_size, target_hidden_size],
        "target_hc_head_shapes_match": (
            target_hc_head_fn_shape == [target_hc_mult, target_hc_mult * target_hidden_size]
            and target_hc_head_base_shape == [target_hc_mult]
            and target_hc_head_scale_shape == [1]
        ),
        "target_safetensors_shards_exist": target_manifest.all_shards_exist(),
        "draft_weight_prefixes_match_runtime": all(draft_manifest.has_weight(key) for key in draft_required_keys)
        and draft_layer_modules_match_runtime,
        "draft_layer_norm_shapes_match": draft_layer_norm_shapes_match,
        "draft_main_projection_shapes_match": (
            draft_main_proj_shape == [draft_hidden_size, target_hidden_size * draft_layer_count]
            and draft_main_norm_shape == [draft_hidden_size]
        ),
        "draft_embedding_shape_matches": draft_embedding_shape == [draft_vocab_size, draft_hidden_size],
        "draft_final_norm_shape_matches": draft_final_norm_shape == [draft_hidden_size],
        "draft_lm_head_shape_matches": draft_lm_head_shape == [draft_vocab_size, draft_hidden_size],
        "draft_hc_head_shapes_match": (
            draft_hc_head_fn_shape == [draft_hc_mult, draft_hc_mult * draft_hidden_size]
            and draft_hc_head_base_shape == [draft_hc_mult]
            and draft_hc_head_scale_shape == [1]
        ),
        "draft_markov_modules_match_runtime": all(draft_manifest.has_weight(key) for key in draft_markov_keys),
        "draft_markov_shapes_match": (
            draft_markov_w1_shape == [draft_vocab_size, draft_markov_rank]
            and draft_markov_w2_shape == [draft_vocab_size, draft_markov_rank]
        ),
        "draft_confidence_module_matches_runtime": (draft_manifest.has_weight(draft_confidence_weight_key)),
        "draft_confidence_shapes_match": (
            draft_confidence_weight_shape == [1, draft_hidden_size + draft_markov_rank]
            and draft_confidence_bias_shape in (None, [1])
        ),
        "draft_safetensors_shards_exist": draft_manifest.all_shards_exist(),
        "hc_mult_matches": target_hc_mult == draft_hc_mult,
    }
    weight_layout = {
        "target_inspected": True,
        "target_model_prefix": "",
        "target_safetensors_shard_count": target_manifest.shard_count(),
        "target_embedding_shape": target_embedding_shape,
        "target_final_norm_shape": target_final_norm_shape,
        "target_lm_head_shape": target_lm_head_shape,
        "target_hc_head_fn_shape": target_hc_head_fn_shape,
        "target_hc_head_base_shape": target_hc_head_base_shape,
        "target_hc_head_scale_shape": target_hc_head_scale_shape,
        "draft_inspected": True,
        "draft_model_prefix": "mtp.",
        "draft_safetensors_shard_count": draft_manifest.shard_count(),
        "draft_uses_own_embedding": True,
        "draft_embedding_source": ("dedicated" if draft_embedding_key == dedicated_embedding_key else "fallback"),
        "draft_uses_own_lm_head": True,
        "draft_lm_head_source": ("dedicated" if draft_lm_head_key == dedicated_lm_head_key else "fallback"),
        "draft_main_proj_shape": draft_main_proj_shape,
        "draft_main_norm_shape": draft_main_norm_shape,
        "draft_embedding_shape": draft_embedding_shape,
        "draft_final_norm_shape": draft_final_norm_shape,
        "draft_lm_head_shape": draft_lm_head_shape,
        "draft_hc_head_fn_shape": draft_hc_head_fn_shape,
        "draft_hc_head_base_shape": draft_hc_head_base_shape,
        "draft_hc_head_scale_shape": draft_hc_head_scale_shape,
        "draft_markov_rank": draft_markov_rank,
        "draft_markov_w1_shape": draft_markov_w1_shape,
        "draft_markov_w2_shape": draft_markov_w2_shape,
        "draft_confidence_weight_shape": draft_confidence_weight_shape,
        "draft_confidence_bias_shape": draft_confidence_bias_shape,
        "draft_layer_count_inspected": draft_layer_count,
        "draft_layer_norm_shapes": draft_layer_norm_shapes,
    }
    return checks, weight_layout


def validate_block_spec_checkpoint_pair(
    target_dir: Path,
    draft_dir: Path,
    algorithm: str,
    num_speculative_tokens: int,
) -> dict[str, Any]:
    """Return explicit startup checks for one Block-Spec checkpoint pair."""
    if algorithm not in {"DFlash", "DSpark"}:
        raise ValueError(f"unsupported Block-Spec algorithm: {algorithm}")
    if num_speculative_tokens <= 0:
        raise ValueError("num_speculative_tokens must be positive")

    target = _load_config(target_dir, "target")
    draft = _load_config(draft_dir, "draft")
    target_model = _model_config(target)
    draft_model = _model_config(draft)

    target_hidden_size = _required_positive_int(target_model, "hidden_size", "target config")
    draft_hidden_size = _required_positive_int(draft_model, "hidden_size", "draft config")
    target_vocab_size = _required_positive_int(target_model, "vocab_size", "target config")
    draft_vocab_size = _required_positive_int(draft_model, "vocab_size", "draft config")
    target_layer_count = _required_positive_int(target_model, "num_hidden_layers", "target config")
    draft_layer_count = _required_positive_int(draft_model, "num_hidden_layers", "draft config")

    capture_layer_ids = _first_value(
        draft,
        (
            ("dspark_target_layer_ids",),
            ("target_layer_ids",),
            ("dflash_config", "target_layer_ids"),
        ),
    )
    capture_layers_are_integers = (
        isinstance(capture_layer_ids, list)
        and bool(capture_layer_ids)
        and all(isinstance(layer_id, int) and not isinstance(layer_id, bool) for layer_id in capture_layer_ids)
    )
    capture_layers_are_unique = capture_layers_are_integers and (len(set(capture_layer_ids)) == len(capture_layer_ids))
    capture_layer_count = len(capture_layer_ids) if isinstance(capture_layer_ids, list) else 0
    capture_layers_are_in_range = capture_layers_are_integers and all(
        0 <= layer_id < target_layer_count for layer_id in capture_layer_ids
    )

    mask_token_id = _first_value(
        draft,
        (
            ("dflash_config", "mask_token_id"),
            ("mask_token_id",),
            ("dspark_noise_token_id",),
        ),
    )
    mask_token_is_valid = (
        isinstance(mask_token_id, int) and not isinstance(mask_token_id, bool) and 0 <= mask_token_id < draft_vocab_size
    )

    declared_target_layers = draft.get("num_target_layers")
    declared_target_layers_match = declared_target_layers is None or (
        isinstance(declared_target_layers, int)
        and not isinstance(declared_target_layers, bool)
        and declared_target_layers == target_layer_count
    )
    checkpoint_block_size = _first_value(draft, (("dspark_block_size",), ("block_size",)))
    checkpoint_block_size_is_valid = checkpoint_block_size is None or (
        isinstance(checkpoint_block_size, int)
        and not isinstance(checkpoint_block_size, bool)
        and checkpoint_block_size > 0
    )
    deepseek_v4_dspark = _is_deepseek_v4_dspark(algorithm, draft_model)
    draft_effective_layer_count = capture_layer_count if deepseek_v4_dspark else draft_layer_count
    draft_layer_count_matches_capture_layers = (
        capture_layers_are_integers and draft_effective_layer_count == capture_layer_count
    )

    checks = {
        "checkpoint_paths_are_distinct": target_dir.resolve() != draft_dir.resolve(),
        "target_weights_present": _has_weights(target_dir),
        "draft_weights_present": _has_weights(draft_dir),
        "draft_architecture_matches_algorithm": _draft_architecture_matches(draft, algorithm),
        "hidden_size_matches": target_hidden_size == draft_hidden_size,
        "vocab_size_matches": target_vocab_size == draft_vocab_size,
        "capture_layers_are_integers": capture_layers_are_integers,
        "capture_layers_are_unique": capture_layers_are_unique,
        "capture_layers_are_in_range": capture_layers_are_in_range,
        "mask_token_is_valid": mask_token_is_valid,
        "declared_target_layers_match": declared_target_layers_match,
        "checkpoint_block_size_is_valid": checkpoint_block_size_is_valid,
        "draft_layer_count_matches_capture_layers": (draft_layer_count_matches_capture_layers),
    }
    weight_layout: dict[str, Any] = {
        "target_inspected": False,
        "draft_inspected": False,
    }
    inspect_qwen_weight_layout = (
        algorithm in {"DFlash", "DSpark"}
        and draft_model.get("model_type") == "qwen3"
        and _draft_architecture_matches(draft, algorithm)
    )
    if inspect_qwen_weight_layout:
        target_model_type = _qwen_target_model_type(target, target_model)
        target_prefix = _QWEN_TARGET_PREFIXES[target_model_type] if target_model_type is not None else ""
        target_manifest = SafetensorsManifest(target_dir, "target")
        target_embedding_key = target_prefix + "embed_tokens.weight"
        target_final_norm_key = target_prefix + "norm.weight"
        target_first_layer_norm_key = target_prefix + "layers.0.input_layernorm.weight"
        target_last_layer_norm_key = target_prefix + f"layers.{target_layer_count - 1}.input_layernorm.weight"
        target_tie_word_embeddings = _required_bool(
            target_model,
            "tie_word_embeddings",
            _required_bool(
                target,
                "tie_word_embeddings",
                False,
                "target config",
            ),
            "target model config",
        )
        target_lm_head_key = target_embedding_key if target_tie_word_embeddings else "lm_head.weight"
        target_embedding_shape = target_manifest.shape(target_embedding_key)
        target_final_norm_shape = target_manifest.shape(target_final_norm_key)
        target_first_layer_norm_shape = target_manifest.shape(target_first_layer_norm_key)
        target_last_layer_norm_shape = target_manifest.shape(target_last_layer_norm_key)
        target_lm_head_shape = target_manifest.shape(target_lm_head_key)

        draft_head_dim = _required_positive_int(draft_model, "head_dim", "draft config")
        draft_attention_head_count = _required_positive_int(draft_model, "num_attention_heads", "draft config")
        draft_kv_head_count = _required_positive_int(draft_model, "num_key_value_heads", "draft config")
        draft_intermediate_size = _required_positive_int(draft_model, "intermediate_size", "draft config")
        draft_query_width = draft_attention_head_count * draft_head_dim
        draft_kv_width = draft_kv_head_count * draft_head_dim
        draft_manifest = SafetensorsManifest(draft_dir, "draft")
        draft_prefix = _qwen_draft_prefix(draft_manifest)
        draft_effective_prefix = draft_prefix or ""
        draft_fc_key = draft_effective_prefix + "fc.weight"
        draft_hidden_norm_key = draft_effective_prefix + "hidden_norm.weight"
        draft_final_norm_key = draft_effective_prefix + "norm.weight"
        draft_fc_shape = draft_manifest.shape(draft_fc_key)
        draft_hidden_norm_shape = draft_manifest.shape(draft_hidden_norm_key)
        draft_final_norm_shape = draft_manifest.shape(draft_final_norm_key)

        draft_layer_modules_match_runtime = True
        draft_attention_shapes_match = True
        draft_mlp_shapes_match = True
        draft_layer_norm_shapes_match = True
        draft_layer_shapes: list[dict[str, Any]] = []
        for layer_index in range(draft_layer_count):
            layer_prefix = draft_effective_prefix + f"layers.{layer_index}."
            layer_keys = {
                "q_proj": layer_prefix + "self_attn.q_proj.weight",
                "k_proj": layer_prefix + "self_attn.k_proj.weight",
                "v_proj": layer_prefix + "self_attn.v_proj.weight",
                "o_proj": layer_prefix + "self_attn.o_proj.weight",
                "q_norm": layer_prefix + "self_attn.q_norm.weight",
                "k_norm": layer_prefix + "self_attn.k_norm.weight",
                "gate_proj": layer_prefix + "mlp.gate_proj.weight",
                "up_proj": layer_prefix + "mlp.up_proj.weight",
                "down_proj": layer_prefix + "mlp.down_proj.weight",
                "input_layernorm": layer_prefix + "input_layernorm.weight",
                "post_attention_layernorm": layer_prefix + "post_attention_layernorm.weight",
            }
            layer_shapes = {name: draft_manifest.shape(key) for name, key in layer_keys.items()}
            draft_layer_modules_match_runtime = draft_layer_modules_match_runtime and all(
                draft_manifest.has_weight(key) for key in layer_keys.values()
            )
            draft_attention_shapes_match = (
                draft_attention_shapes_match
                and layer_shapes["q_proj"] == [draft_query_width, draft_hidden_size]
                and layer_shapes["k_proj"] == [draft_kv_width, draft_hidden_size]
                and layer_shapes["v_proj"] == [draft_kv_width, draft_hidden_size]
                and layer_shapes["o_proj"] == [draft_hidden_size, draft_query_width]
                and layer_shapes["q_norm"] == [draft_head_dim]
                and layer_shapes["k_norm"] == [draft_head_dim]
            )
            draft_mlp_shapes_match = (
                draft_mlp_shapes_match
                and layer_shapes["gate_proj"] == [draft_intermediate_size, draft_hidden_size]
                and layer_shapes["up_proj"] == [draft_intermediate_size, draft_hidden_size]
                and layer_shapes["down_proj"] == [draft_hidden_size, draft_intermediate_size]
            )
            draft_layer_norm_shapes_match = (
                draft_layer_norm_shapes_match
                and layer_shapes["input_layernorm"] == [draft_hidden_size]
                and layer_shapes["post_attention_layernorm"] == [draft_hidden_size]
            )
            draft_layer_shapes.append({"layer_index": layer_index, **layer_shapes})

        checks.update(
            {
                "target_runtime_model_type_supported": (target_model_type is not None),
                "target_weight_prefixes_match_runtime": (
                    target_model_type is not None
                    and all(
                        target_manifest.has_weight(key)
                        for key in (
                            target_embedding_key,
                            target_final_norm_key,
                            target_first_layer_norm_key,
                            target_last_layer_norm_key,
                            target_lm_head_key,
                        )
                    )
                ),
                "target_embedding_shape_matches": target_embedding_shape == [target_vocab_size, target_hidden_size],
                "target_final_norm_shape_matches": target_final_norm_shape == [target_hidden_size],
                "target_lm_head_shape_matches": target_lm_head_shape == [target_vocab_size, target_hidden_size],
                "target_boundary_layer_norm_shapes_match": (
                    target_first_layer_norm_shape == [target_hidden_size]
                    and target_last_layer_norm_shape == [target_hidden_size]
                ),
                "target_safetensors_shards_exist": (target_manifest.all_shards_exist()),
                "draft_weight_prefix_matches_runtime": draft_prefix is not None,
                "draft_required_modules_match_runtime": (
                    draft_prefix is not None
                    and all(
                        draft_manifest.has_weight(key)
                        for key in (
                            draft_fc_key,
                            draft_hidden_norm_key,
                            draft_final_norm_key,
                        )
                    )
                    and draft_layer_modules_match_runtime
                ),
                "draft_fc_shape_matches_capture_width": draft_fc_shape
                == [
                    draft_hidden_size,
                    target_hidden_size * capture_layer_count,
                ],
                "draft_norm_shapes_match": (
                    draft_hidden_norm_shape == [draft_hidden_size] and draft_final_norm_shape == [draft_hidden_size]
                ),
                "draft_attention_shapes_match": (draft_attention_shapes_match),
                "draft_mlp_shapes_match": draft_mlp_shapes_match,
                "draft_layer_norm_shapes_match": (draft_layer_norm_shapes_match),
                "draft_safetensors_shards_exist": (draft_manifest.all_shards_exist()),
            }
        )
        weight_layout.update(
            {
                "target_inspected": True,
                "target_model_type": target_model_type,
                "target_model_prefix": target_prefix,
                "target_tie_word_embeddings": target_tie_word_embeddings,
                "target_lm_head_source": ("embedding" if target_tie_word_embeddings else "lm_head"),
                "target_safetensors_shard_count": (target_manifest.shard_count()),
                "target_embedding_shape": target_embedding_shape,
                "target_final_norm_shape": target_final_norm_shape,
                "target_first_layer_norm_shape": (target_first_layer_norm_shape),
                "target_last_layer_norm_shape": target_last_layer_norm_shape,
                "target_lm_head_shape": target_lm_head_shape,
                "draft_inspected": True,
                "draft_model_prefix": draft_prefix,
                "draft_safetensors_shard_count": (draft_manifest.shard_count()),
                "draft_uses_target_shared_embedding": True,
                "draft_uses_target_shared_lm_head": True,
                "draft_fc_shape": draft_fc_shape,
                "draft_hidden_norm_shape": draft_hidden_norm_shape,
                "draft_final_norm_shape": draft_final_norm_shape,
                "draft_layer_count_inspected": draft_layer_count,
                "draft_layer_shapes": draft_layer_shapes,
            }
        )
        if algorithm == "DSpark":
            draft_markov_rank = _required_positive_int(draft_model, "markov_rank", "draft config")
            confidence_head_enabled = _required_bool(
                draft_model,
                "enable_confidence_head",
                False,
                "draft config",
            )
            confidence_head_with_markov = _required_bool(
                draft_model,
                "confidence_head_with_markov",
                False,
                "draft config",
            )
            draft_markov_w1_key = draft_effective_prefix + "markov_head.markov_w1.weight"
            draft_markov_w2_key = draft_effective_prefix + "markov_head.markov_w2.weight"
            draft_markov_w1_shape = draft_manifest.shape(draft_markov_w1_key)
            draft_markov_w2_shape = draft_manifest.shape(draft_markov_w2_key)
            draft_confidence_weight_key = draft_effective_prefix + "confidence_head.proj.weight"
            draft_confidence_bias_key = draft_effective_prefix + "confidence_head.proj.bias"
            draft_confidence_weight_shape = draft_manifest.shape(draft_confidence_weight_key)
            draft_confidence_bias_shape = draft_manifest.shape(draft_confidence_bias_key)
            draft_confidence_input_width = draft_hidden_size
            if confidence_head_with_markov:
                draft_confidence_input_width += draft_markov_rank
            checks.update(
                {
                    "draft_markov_modules_match_runtime": all(
                        draft_manifest.has_weight(key)
                        for key in (
                            draft_markov_w1_key,
                            draft_markov_w2_key,
                        )
                    ),
                    "draft_markov_shapes_match": (
                        draft_markov_w1_shape == [draft_vocab_size, draft_markov_rank]
                        and draft_markov_w2_shape == [draft_vocab_size, draft_markov_rank]
                    ),
                    "draft_confidence_module_matches_runtime": (
                        not confidence_head_enabled or draft_manifest.has_weight(draft_confidence_weight_key)
                    ),
                    "draft_confidence_shapes_match": (
                        not confidence_head_enabled
                        or (
                            draft_confidence_weight_shape == [1, draft_confidence_input_width]
                            and draft_confidence_bias_shape in (None, [1])
                        )
                    ),
                }
            )
            weight_layout.update(
                {
                    "draft_markov_rank": draft_markov_rank,
                    "draft_markov_w1_shape": draft_markov_w1_shape,
                    "draft_markov_w2_shape": draft_markov_w2_shape,
                    "draft_confidence_head_enabled": (confidence_head_enabled),
                    "draft_confidence_head_with_markov": (confidence_head_with_markov),
                    "draft_confidence_weight_shape": (draft_confidence_weight_shape),
                    "draft_confidence_bias_shape": (draft_confidence_bias_shape),
                }
            )
    elif deepseek_v4_dspark:
        deepseek_checks, weight_layout = _deepseek_v4_dspark_weight_contract(
            target_dir,
            draft_dir,
            target_model,
            draft_model,
            target_hidden_size,
            draft_hidden_size,
            target_vocab_size,
            draft_vocab_size,
            target_layer_count,
            draft_effective_layer_count,
        )
        checks.update(deepseek_checks)
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "passed": not failures,
        "algorithm": algorithm,
        "target_dir": str(target_dir),
        "draft_dir": str(draft_dir),
        "num_speculative_tokens": num_speculative_tokens,
        "checkpoint_block_size": checkpoint_block_size,
        "target_hidden_size": target_hidden_size,
        "draft_hidden_size": draft_hidden_size,
        "target_vocab_size": target_vocab_size,
        "draft_vocab_size": draft_vocab_size,
        "target_layer_count": target_layer_count,
        "draft_layer_count": draft_layer_count,
        "draft_effective_layer_count": draft_effective_layer_count,
        "capture_layer_ids": capture_layer_ids,
        "mask_token_id": mask_token_id,
        "weight_layout": weight_layout,
        "checks": checks,
        "failures": failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--algorithm", required=True, choices=("DFlash", "DSpark"))
    parser.add_argument("--num-speculative-tokens", required=True, type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = validate_block_spec_checkpoint_pair(
            args.target,
            args.draft,
            args.algorithm,
            args.num_speculative_tokens,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        logger.exception("failed to validate Block-Spec checkpoint pair")
        return 2

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    if not result["passed"]:
        logger.error(f"Block-Spec checkpoint validation failed: {result['failures']}")
        return 1
    logger.info("Block-Spec checkpoint validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
