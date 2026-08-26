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

"""Read safetensors index and header metadata without loading tensor payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SafetensorsManifest:
    """Resolve safetensors keys, shards, and shapes for one checkpoint."""

    def __init__(self, model_dir: Path, label: str) -> None:
        self._model_dir = model_dir
        self._label = label
        self._header_cache: dict[Path, dict[str, Any]] = {}
        self._shard_file_names: set[str] = set()
        self._weight_files = self._load_weight_files()

    def _load_weight_files(self) -> dict[str, str]:
        index_paths = sorted(self._model_dir.glob("*.safetensors.index.json"))
        weight_files: dict[str, str] = {}
        indexed_file_names: set[str] = set()
        for index_path in index_paths:
            with index_path.open("r", encoding="utf-8") as index_file:
                index = json.load(index_file)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"{self._label} safetensors index has no weight_map: {index_path}")
            for key, file_name in weight_map.items():
                if not isinstance(key, str) or not isinstance(file_name, str):
                    raise ValueError(
                        f"{self._label} safetensors index contains an invalid weight mapping: {index_path}"
                    )
                if Path(file_name).name != file_name:
                    raise ValueError(
                        f"{self._label} safetensors shard must be in the checkpoint directory: {file_name}"
                    )
                previous_file = weight_files.get(key)
                if previous_file is not None and previous_file != file_name:
                    raise ValueError(f"{self._label} weight is mapped to multiple shards: {key}")
                weight_files[key] = file_name
                indexed_file_names.add(file_name)
                self._shard_file_names.add(file_name)

        weight_paths = sorted(self._model_dir.glob("*.safetensors"))
        for weight_path in weight_paths:
            if weight_path.name in indexed_file_names:
                continue
            header = self._read_header(weight_path)
            for key in header:
                if key == "__metadata__":
                    continue
                if key in weight_files:
                    raise ValueError(f"{self._label} weight appears in multiple shards: {key}")
                weight_files[key] = weight_path.name
            self._shard_file_names.add(weight_path.name)
        return weight_files

    def _read_header(self, weight_path: Path) -> dict[str, Any]:
        cached_header = self._header_cache.get(weight_path)
        if cached_header is not None:
            return cached_header
        with weight_path.open("rb") as weight_file:
            header_size_bytes = weight_file.read(8)
            if len(header_size_bytes) != 8:
                raise ValueError(f"{self._label} safetensors header is truncated: {weight_path}")
            header_size = int.from_bytes(header_size_bytes, "little")
            file_size = weight_path.stat().st_size
            if header_size <= 0 or header_size > file_size - 8:
                raise ValueError(f"{self._label} safetensors header size is invalid: {weight_path}")
            header_bytes = weight_file.read(header_size)
        header = json.loads(header_bytes)
        if not isinstance(header, dict):
            raise ValueError(f"{self._label} safetensors header must be an object: {weight_path}")
        self._header_cache[weight_path] = header
        return header

    def has_weight(self, key: str) -> bool:
        return key in self._weight_files

    def has_prefix(self, prefix: str) -> bool:
        return any(key.startswith(prefix) for key in self._weight_files)

    def all_shards_exist(self) -> bool:
        return all((self._model_dir / file_name).is_file() for file_name in self._shard_file_names)

    def shard_count(self) -> int:
        return len(self._shard_file_names)

    def shape(self, key: str) -> list[int] | None:
        file_name = self._weight_files.get(key)
        if file_name is None:
            return None
        weight_path = self._model_dir / file_name
        if not weight_path.is_file():
            raise ValueError(f"{self._label} safetensors shard does not exist: {weight_path}")
        tensor_metadata = self._read_header(weight_path).get(key)
        if not isinstance(tensor_metadata, dict):
            raise ValueError(f"{self._label} safetensors shard has no metadata for weight: {key}")
        shape = tensor_metadata.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
        ):
            raise ValueError(f"{self._label} safetensors weight has an invalid shape: {key}")
        return shape
