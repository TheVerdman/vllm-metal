# SPDX-License-Identifier: Apache-2.0
"""Temporary Nano VL V2 config alias for the pinned mlx-vlm loader."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_metal.v1.mlx_vlm_paths import mlx_vlm_compatible_model_path


def _write_config(model_path: Path, *, model_type: str, architecture: str) -> dict:
    config = {
        "model_type": model_type,
        "architectures": [architecture],
        "llm_config": {"model_type": "nemotron_h"},
        "vision_config": {"model_type": "radio"},
        "img_context_token_id": 131072,
    }
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return config


def test_exact_nano_checkpoint_gets_temporary_loader_alias(tmp_path: Path) -> None:
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    config = _write_config(
        model_path,
        model_type="NemotronH_Nano_VL_V2",
        architecture="NemotronH_Nano_VL_V2",
    )
    weights = model_path / "model.safetensors"
    weights.write_bytes(b"fake-weight-file")
    tokenizer = model_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")

    with mlx_vlm_compatible_model_path(model_path) as compatible:
        assert compatible != model_path
        assert (compatible / "model.safetensors").is_symlink()
        assert (compatible / "model.safetensors").resolve() == weights
        assert (compatible / "tokenizer.json").resolve() == tokenizer
        translated = json.loads((compatible / "config.json").read_text())
        assert translated == {
            **config,
            "model_type": "nemotron_h_nano_omni",
            "vision_config": {
                **config["vision_config"],
                "separate_video_embedder": False,
            },
        }

    assert not compatible.exists()
    assert json.loads((model_path / "config.json").read_text()) == config
    assert weights.read_bytes() == b"fake-weight-file"


def test_image_only_patch_projection_loads_strictly_and_preserves_omni() -> None:
    from mlx.utils import tree_flatten
    from mlx_vlm.models.nemotron_h_nano_omni.config import VisionConfig
    from mlx_vlm.models.nemotron_h_nano_omni.vision import ViTPatchGenerator

    from vllm_metal.v1.mlx_vlm_paths import _respect_nano_image_only_vision_config

    _respect_nano_image_only_vision_config()
    image_config = VisionConfig(
        hidden_size=16,
        image_size=32,
        max_resolution=32,
        separate_video_embedder=False,
    )
    image_model = ViTPatchGenerator(image_config)
    assert not hasattr(image_model, "video_embedder")
    weights = tree_flatten(image_model.parameters())
    image_model.load_weights(weights, strict=True)

    omni_model = ViTPatchGenerator(
        VisionConfig(
            hidden_size=16,
            image_size=32,
            max_resolution=32,
            separate_video_embedder=True,
        )
    )
    assert hasattr(omni_model, "video_embedder")
    with pytest.raises(ValueError, match="video_embedder.weight"):
        omni_model.load_weights(weights, strict=True)


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("nemotron_h_nano_omni", "NemotronH_Nano_VL_V2"),
        ("NemotronH_Nano_VL_V2", "AnotherArchitecture"),
        ("qwen3_vl", "Qwen3VLForConditionalGeneration"),
    ],
)
def test_other_configs_are_left_unchanged(
    tmp_path: Path, model_type: str, architecture: str
) -> None:
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    _write_config(model_path, model_type=model_type, architecture=architecture)

    with mlx_vlm_compatible_model_path(model_path) as compatible:
        assert compatible == model_path


def test_other_remote_models_keep_the_normal_loader_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_vlm import utils as mlx_vlm_utils

    def unexpected_download(*args: Any, **kwargs: Any) -> None:
        pytest.fail("The Nano alias must not resolve unrelated model repositories")

    monkeypatch.setattr(mlx_vlm_utils, "get_model_path", unexpected_download)
    config = SimpleNamespace(model_type="qwen3_vl", architectures=["Qwen3VL"])
    with mlx_vlm_compatible_model_path(
        "example/qwen", hf_config=config, revision="pinned-revision"
    ) as compatible:
        assert str(compatible) == "example/qwen"


def test_pinned_mlx_vlm_imports_translated_nano_model(tmp_path: Path) -> None:
    pytest.importorskip("mlx_vlm.models.nemotron_h.language")
    from mlx_vlm.utils import get_model_and_args

    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    config = _write_config(
        model_path,
        model_type="NemotronH_Nano_VL_V2",
        architecture="NemotronH_Nano_VL_V2",
    )
    with pytest.raises(ValueError, match="not supported"):
        get_model_and_args(config)

    with mlx_vlm_compatible_model_path(model_path) as compatible:
        translated = json.loads((compatible / "config.json").read_text())
        module, model_type = get_model_and_args(translated)
        assert model_type == "nemotron_h_nano_omni"
        assert module.Model.__name__ == "Model"


@pytest.mark.parametrize("from_hub", [False, True])
@pytest.mark.parametrize("lazy_weights", [False, True])
def test_lifecycle_passes_compatible_path_to_mlx_vlm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    from_hub: bool,
    lazy_weights: bool,
) -> None:
    from mlx_vlm import utils as mlx_vlm_utils

    from vllm_metal.v1 import model_lifecycle

    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    config = _write_config(
        model_path,
        model_type="NemotronH_Nano_VL_V2",
        architecture="NemotronH_Nano_VL_V2",
    )
    seen: dict[str, Any] = {}
    model = object()
    tokenizer = object()

    def fake_get_model_path(name: str, *, revision: str | None) -> Path:
        assert from_hub
        assert name == "example/nano-vl"
        assert revision == "pinned-revision"
        seen["resolved"] = True
        return model_path

    def fake_mlx_vlm_load(path: str, *, lazy: bool, revision: str | None):
        seen["path"] = Path(path)
        seen["model_type"] = json.loads((Path(path) / "config.json").read_text())[
            "model_type"
        ]
        seen["lazy"] = lazy
        seen["revision"] = revision
        return model, tokenizer

    monkeypatch.setattr(model_lifecycle, "mlx_vlm_load", fake_mlx_vlm_load)
    monkeypatch.setattr(mlx_vlm_utils, "get_model_path", fake_get_model_path)
    lifecycle = model_lifecycle.ModelLifecycle.__new__(model_lifecycle.ModelLifecycle)
    loaded = lifecycle._load_generation_model(
        "example/nano-vl" if from_hub else str(model_path),
        True,
        model_config=SimpleNamespace(
            revision="pinned-revision", hf_config=SimpleNamespace(**config)
        ),
        tokenizer_config={},
        lazy_weights=lazy_weights,
    )

    assert loaded == (model, tokenizer)
    assert seen["path"] != model_path
    assert seen["model_type"] == "nemotron_h_nano_omni"
    assert seen.get("resolved", False) is from_hub
    assert seen["lazy"] is lazy_weights
    assert seen["revision"] == "pinned-revision"
    assert not seen["path"].exists()
