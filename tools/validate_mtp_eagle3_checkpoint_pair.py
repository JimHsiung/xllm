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

"""Validate a target/draft checkpoint pair for xLLM MTP or Eagle3."""

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

_MTP_RUNTIME_MODEL_TYPES = {
    "deepseek_v3_mtp",
    "deepseek_v32_mtp",
    "deepseek_v4_mtp",
    "glm4_moe_mtp",
    "glm_moe_dsa_mtp",
    "joyai_llm_flash_mtp",
    "mimo_mtp",
    "qwen3_5_moe_mtp",
    "qwen3_5_mtp",
}

_EAGLE3_RUNTIME_MODEL_TYPES = {
    "kimi_k25_eagle3",
    "qwen3_eagle3",
}

_EAGLE3_TARGET_RUNTIME_MODEL_TYPES = {
    "kimi_k25",
    "qwen3",
    "qwen3_moe",
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


def _optional_nonnegative_int(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer when present")
    return value


def _positive_int_or_default(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    key: str,
    default: int,
    context: str,
) -> int:
    value = primary.get(key, secondary.get(key, default))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer")
    return value


def _required_model_type(config: dict[str, Any], context: str) -> str:
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError(f"{context}.model_type must be a non-empty string")
    return model_type


def _mtp_layer_count(primary: dict[str, Any], secondary: dict[str, Any]) -> int | None:
    for config in (primary, secondary):
        for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
            if config.get(key) is not None:
                return _optional_nonnegative_int(config, key)
    return None


def _has_weights(model_dir: Path) -> bool:
    return any(path.is_file() for pattern in _WEIGHT_PATTERNS for path in model_dir.glob(pattern))


def _architecture_matches(config: dict[str, Any], algorithm: str) -> bool:
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        return False
    marker = "mtp" if algorithm == "MTP" else "eagle3"
    return any(isinstance(architecture, str) and marker in architecture.lower() for architecture in architectures)


def _mtp_model_family_matches(
    target: dict[str, Any],
    target_model: dict[str, Any],
    draft_model_type: str,
) -> bool:
    expected_types: set[str] = set()
    for config in (target, target_model):
        target_model_type = config.get("model_type")
        if not isinstance(target_model_type, str) or not target_model_type:
            continue
        expected_types.add(f"{target_model_type}_mtp")
        if target_model_type.endswith("_text"):
            expected_types.add(f"{target_model_type[:-5]}_mtp")
    return draft_model_type in expected_types


def _optional_token_matches(
    target: dict[str, Any],
    target_model: dict[str, Any],
    draft: dict[str, Any],
    token_key: str,
) -> bool:
    target_token = target.get(token_key, target_model.get(token_key))
    draft_token = draft.get(token_key)
    if target_token is None or draft_token is None:
        return True
    return (
        isinstance(target_token, int)
        and not isinstance(target_token, bool)
        and isinstance(draft_token, int)
        and not isinstance(draft_token, bool)
        and target_token == draft_token
    )


def _eagle3_capture_layer_ids(target_layer_count: int) -> list[int]:
    return [2, target_layer_count // 2, target_layer_count - 3]


def validate_mtp_eagle3_checkpoint_pair(
    target_dir: Path,
    draft_dir: Path,
    algorithm: str,
    num_speculative_tokens: int,
) -> dict[str, Any]:
    """Return explicit startup checks for one MTP or Eagle3 pair."""
    if algorithm not in {"MTP", "Eagle3"}:
        raise ValueError(f"unsupported speculative algorithm: {algorithm}")
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
    target_model_type = _required_model_type(target, "target config")
    draft_model_type = _required_model_type(draft, "draft config")

    draft_effective_vocab_size = _optional_nonnegative_int(draft_model, "draft_vocab_size")
    if draft_effective_vocab_size is None:
        draft_effective_vocab_size = draft_vocab_size

    checks = {
        "checkpoint_paths_are_distinct": target_dir.resolve() != draft_dir.resolve(),
        "target_weights_present": _has_weights(target_dir),
        "draft_weights_present": _has_weights(draft_dir),
        "draft_architecture_matches_algorithm": _architecture_matches(draft, algorithm),
        "hidden_size_matches": target_hidden_size == draft_hidden_size,
        "bos_token_id_matches": _optional_token_matches(target, target_model, draft, "bos_token_id"),
        "eos_token_id_matches": _optional_token_matches(target, target_model, draft, "eos_token_id"),
    }

    target_mtp_layer_count: int | None = None
    draft_declared_mtp_layer_count: int | None = None
    target_hc_mult: int | None = None
    draft_hc_mult: int | None = None
    target_capture_layer_ids: list[int] = []
    weight_layout: dict[str, Any] = {
        "target_inspected": False,
        "draft_inspected": False,
    }
    if algorithm == "MTP":
        target_mtp_layer_count = _mtp_layer_count(target_model, target)
        draft_declared_mtp_layer_count = _mtp_layer_count(draft_model, draft)
        checks.update(
            {
                "draft_runtime_model_type_supported": draft_model_type in _MTP_RUNTIME_MODEL_TYPES,
                "model_family_matches": _mtp_model_family_matches(target, target_model, draft_model_type),
                "vocab_size_matches": target_vocab_size == draft_effective_vocab_size,
                "target_declares_mtp_layers": target_mtp_layer_count is not None and target_mtp_layer_count > 0,
                "draft_layer_count_matches_target_mtp_layers": (
                    target_mtp_layer_count is not None
                    and target_mtp_layer_count > 0
                    and draft_layer_count == target_mtp_layer_count
                ),
                "draft_declared_mtp_layers_match": (
                    draft_declared_mtp_layer_count is None or draft_declared_mtp_layer_count == target_mtp_layer_count
                ),
            }
        )
        if target_model_type == "deepseek_v4":
            target_hc_mult = _positive_int_or_default(
                target_model,
                target,
                "hc_mult",
                4,
                "target config",
            )
            target_manifest = SafetensorsManifest(target_dir, "target")
            target_last_layer_prefix = f"layers.{target_layer_count - 1}."
            target_embedding_shape = target_manifest.shape("embed.weight")
            target_final_norm_shape = target_manifest.shape("norm.weight")
            target_lm_head_shape = target_manifest.shape("head.weight")
            target_hc_head_fn_shape = target_manifest.shape("hc_head_fn")
            target_hc_head_base_shape = target_manifest.shape("hc_head_base")
            target_hc_head_scale_shape = target_manifest.shape("hc_head_scale")
            checks.update(
                {
                    "target_weight_prefixes_match_runtime": all(
                        target_manifest.has_weight(key)
                        for key in (
                            "embed.weight",
                            "norm.weight",
                            "head.weight",
                            "hc_head_fn",
                            "hc_head_base",
                            "hc_head_scale",
                        )
                    )
                    and target_manifest.has_prefix("layers.0.")
                    and target_manifest.has_prefix(target_last_layer_prefix),
                    "target_embedding_shape_matches": target_embedding_shape == [target_vocab_size, target_hidden_size],
                    "target_final_norm_shape_matches": target_final_norm_shape == [target_hidden_size],
                    "target_lm_head_shape_matches": target_lm_head_shape == [target_vocab_size, target_hidden_size],
                    "target_hc_head_shapes_match": (
                        target_hc_head_fn_shape
                        == [
                            target_hc_mult,
                            target_hc_mult * target_hidden_size,
                        ]
                        and target_hc_head_base_shape == [target_hc_mult]
                        and target_hc_head_scale_shape == [1]
                    ),
                    "target_safetensors_shards_exist": (target_manifest.all_shards_exist()),
                }
            )
            weight_layout["target_inspected"] = True
            weight_layout["target_model_prefix"] = ""
            weight_layout["target_safetensors_shard_count"] = target_manifest.shard_count()
            weight_layout["target_embedding_shape"] = target_embedding_shape
            weight_layout["target_final_norm_shape"] = target_final_norm_shape
            weight_layout["target_lm_head_shape"] = target_lm_head_shape
            weight_layout["target_hc_head_fn_shape"] = target_hc_head_fn_shape
            weight_layout["target_hc_head_base_shape"] = target_hc_head_base_shape
            weight_layout["target_hc_head_scale_shape"] = target_hc_head_scale_shape

        if draft_model_type == "deepseek_v4_mtp":
            draft_hc_mult = _positive_int_or_default(
                draft_model,
                draft,
                "hc_mult",
                4,
                "draft config",
            )
            draft_manifest = SafetensorsManifest(draft_dir, "draft")
            draft_layer_prefix = "model.layers.0."
            draft_enorm_shape = draft_manifest.shape(draft_layer_prefix + "enorm.weight")
            draft_hnorm_shape = draft_manifest.shape(draft_layer_prefix + "hnorm.weight")
            draft_e_proj_shape = draft_manifest.shape(draft_layer_prefix + "e_proj.weight")
            draft_h_proj_shape = draft_manifest.shape(draft_layer_prefix + "h_proj.weight")
            draft_final_norm_shape = draft_manifest.shape(draft_layer_prefix + "norm.weight")
            draft_hc_head_fn_shape = draft_manifest.shape(draft_layer_prefix + "hc_head_fn")
            draft_hc_head_base_shape = draft_manifest.shape(draft_layer_prefix + "hc_head_base")
            draft_hc_head_scale_shape = draft_manifest.shape(draft_layer_prefix + "hc_head_scale")
            draft_required_keys = (
                "enorm.weight",
                "hnorm.weight",
                "e_proj.weight",
                "h_proj.weight",
                "norm.weight",
                "attn_norm.weight",
                "ffn_norm.weight",
                "hc_head_fn",
                "hc_head_base",
                "hc_head_scale",
            )
            checks.update(
                {
                    "draft_weight_prefixes_match_runtime": all(
                        draft_manifest.has_weight(draft_layer_prefix + key) for key in draft_required_keys
                    )
                    and draft_manifest.has_prefix(draft_layer_prefix + "attn.")
                    and draft_manifest.has_prefix(draft_layer_prefix + "ffn."),
                    "draft_norm_shapes_match": (
                        draft_enorm_shape == [draft_hidden_size]
                        and draft_hnorm_shape == [draft_hidden_size]
                        and draft_final_norm_shape == [draft_hidden_size]
                    ),
                    "draft_projection_shapes_match": (
                        draft_e_proj_shape == [draft_hidden_size, draft_hidden_size]
                        and draft_h_proj_shape == [draft_hidden_size, draft_hidden_size]
                    ),
                    "draft_hc_head_shapes_match": (
                        draft_hc_head_fn_shape
                        == [
                            draft_hc_mult,
                            draft_hc_mult * draft_hidden_size,
                        ]
                        and draft_hc_head_base_shape == [draft_hc_mult]
                        and draft_hc_head_scale_shape == [1]
                    ),
                    "draft_safetensors_shards_exist": (draft_manifest.all_shards_exist()),
                }
            )
            weight_layout["draft_inspected"] = True
            weight_layout["draft_model_prefix"] = "model."
            weight_layout["draft_safetensors_shard_count"] = draft_manifest.shard_count()
            weight_layout["draft_uses_target_shared_embedding"] = True
            weight_layout["draft_uses_target_shared_lm_head"] = True
            weight_layout["draft_enorm_shape"] = draft_enorm_shape
            weight_layout["draft_hnorm_shape"] = draft_hnorm_shape
            weight_layout["draft_e_proj_shape"] = draft_e_proj_shape
            weight_layout["draft_h_proj_shape"] = draft_h_proj_shape
            weight_layout["draft_final_norm_shape"] = draft_final_norm_shape
            weight_layout["draft_hc_head_fn_shape"] = draft_hc_head_fn_shape
            weight_layout["draft_hc_head_base_shape"] = draft_hc_head_base_shape
            weight_layout["draft_hc_head_scale_shape"] = draft_hc_head_scale_shape
        if target_hc_mult is not None and draft_hc_mult is not None:
            checks["hc_mult_matches"] = target_hc_mult == draft_hc_mult
    else:
        target_capture_layer_ids = _eagle3_capture_layer_ids(target_layer_count)
        capture_layers_are_unique = len(set(target_capture_layer_ids)) == len(target_capture_layer_ids)
        capture_layers_are_in_range = all(0 <= layer_id < target_layer_count for layer_id in target_capture_layer_ids)
        checks.update(
            {
                "target_runtime_model_type_supported": target_model_type in _EAGLE3_TARGET_RUNTIME_MODEL_TYPES,
                "draft_runtime_model_type_supported": draft_model_type in _EAGLE3_RUNTIME_MODEL_TYPES,
                "draft_layer_count_is_one": draft_layer_count == 1,
                "draft_vocab_is_target_subset": 0 < draft_effective_vocab_size <= target_vocab_size,
                "target_capture_layers_are_unique": capture_layers_are_unique,
                "target_capture_layers_are_in_range": capture_layers_are_in_range,
            }
        )
        if target_model_type == "kimi_k25":
            target_manifest = SafetensorsManifest(target_dir, "target")
            target_embedding_shape = target_manifest.shape("language_model.model.embed_tokens.weight")
            target_final_norm_shape = target_manifest.shape("language_model.model.norm.weight")
            target_lm_head_shape = target_manifest.shape("language_model.lm_head.weight")
            target_required_keys = (
                "language_model.model.embed_tokens.weight",
                "language_model.model.norm.weight",
                "language_model.lm_head.weight",
            )
            checks.update(
                {
                    "target_weight_prefixes_match_runtime": all(
                        target_manifest.has_weight(key) for key in target_required_keys
                    )
                    and target_manifest.has_prefix("language_model.model.layers."),
                    "target_embedding_shape_matches": target_embedding_shape == [target_vocab_size, target_hidden_size],
                    "target_final_norm_shape_matches": target_final_norm_shape == [target_hidden_size],
                    "target_lm_head_shape_matches": target_lm_head_shape == [target_vocab_size, target_hidden_size],
                    "target_safetensors_shards_exist": (target_manifest.all_shards_exist()),
                }
            )
            weight_layout["target_inspected"] = True
            weight_layout["target_safetensors_shard_count"] = target_manifest.shard_count()
            weight_layout["target_embedding_shape"] = target_embedding_shape
            weight_layout["target_final_norm_shape"] = target_final_norm_shape
            weight_layout["target_lm_head_shape"] = target_lm_head_shape

        if draft_model_type == "kimi_k25_eagle3":
            draft_manifest = SafetensorsManifest(draft_dir, "draft")
            draft_fc_shape = draft_manifest.shape("fc.weight")
            draft_final_norm_shape = draft_manifest.shape("norm.weight")
            draft_lm_head_shape = draft_manifest.shape("lm_head.weight")
            draft_d2t_shape = draft_manifest.shape("d2t")
            draft_required_keys = (
                "fc.weight",
                "norm.weight",
                "lm_head.weight",
            )
            checks.update(
                {
                    "draft_weight_modules_match_runtime": all(
                        draft_manifest.has_weight(key) for key in draft_required_keys
                    )
                    and draft_manifest.has_prefix("midlayer."),
                    "draft_fc_shape_matches_3h": draft_fc_shape == [draft_hidden_size, target_hidden_size * 3],
                    "draft_final_norm_shape_matches": draft_final_norm_shape == [draft_hidden_size],
                    "draft_lm_head_shape_matches": draft_lm_head_shape
                    == [draft_effective_vocab_size, draft_hidden_size],
                    "draft_token_mapping_matches_vocab": (
                        draft_d2t_shape == [draft_effective_vocab_size]
                        or (draft_d2t_shape is None and draft_effective_vocab_size == target_vocab_size)
                    ),
                    "draft_safetensors_shards_exist": (draft_manifest.all_shards_exist()),
                }
            )
            weight_layout["draft_inspected"] = True
            weight_layout["draft_safetensors_shard_count"] = draft_manifest.shard_count()
            weight_layout["draft_fc_shape"] = draft_fc_shape
            weight_layout["draft_final_norm_shape"] = draft_final_norm_shape
            weight_layout["draft_lm_head_shape"] = draft_lm_head_shape
            weight_layout["draft_d2t_shape"] = draft_d2t_shape

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "passed": not failures,
        "algorithm": algorithm,
        "target_dir": str(target_dir),
        "draft_dir": str(draft_dir),
        "num_speculative_tokens": num_speculative_tokens,
        "target_hidden_size": target_hidden_size,
        "draft_hidden_size": draft_hidden_size,
        "target_vocab_size": target_vocab_size,
        "draft_vocab_size": draft_vocab_size,
        "draft_effective_vocab_size": draft_effective_vocab_size,
        "target_layer_count": target_layer_count,
        "draft_layer_count": draft_layer_count,
        "target_mtp_layer_count": target_mtp_layer_count,
        "draft_declared_mtp_layer_count": draft_declared_mtp_layer_count,
        "target_hc_mult": target_hc_mult,
        "draft_hc_mult": draft_hc_mult,
        "target_capture_layer_ids": target_capture_layer_ids,
        "required_target_backend": "llm",
        "target_model_type": target_model_type,
        "draft_model_type": draft_model_type,
        "weight_layout": weight_layout,
        "checks": checks,
        "failures": failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--algorithm", required=True, choices=("MTP", "Eagle3"))
    parser.add_argument("--num-speculative-tokens", required=True, type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = validate_mtp_eagle3_checkpoint_pair(
            args.target,
            args.draft,
            args.algorithm,
            args.num_speculative_tokens,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        logger.exception("failed to validate MTP/Eagle3 checkpoint pair")
        return 2

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    if not result["passed"]:
        logger.error(f"MTP/Eagle3 checkpoint validation failed: {result['failures']}")
        return 1
    logger.info("MTP/Eagle3 checkpoint validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
