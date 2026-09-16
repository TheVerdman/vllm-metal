# SPDX-License-Identifier: Apache-2.0
"""Narrow, temporary checkpoint-config view for mlx-vlm loader aliases."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_NANO_VL_ARCHITECTURE = "NemotronH_Nano_VL_V2"
_NANO_VL_MLX_VLM_MODEL_TYPE = "nemotron_h_nano_omni"


def _respect_nano_image_only_vision_config() -> None:
    """Make MLX-VLM 0.6.8 honor its existing separate-video-embedder flag."""
    from mlx_vlm.models.nemotron_h_nano_omni.vision import ViTPatchGenerator

    original = ViTPatchGenerator.__init__
    if getattr(original, "_metal_image_only_guard", False):
        return

    @wraps(original)
    def initialize(self: Any, config: Any) -> None:
        original(self, config)
        # VL V2 has only the image patch projection. MLX-VLM's Omni class
        # otherwise adds an untrained video projection, breaking strict load.
        # Preserve ordinary Omni construction when its flag is enabled.
        if not config.separate_video_embedder and hasattr(self, "video_embedder"):
            del self.video_embedder

    initialize._metal_image_only_guard = True  # type: ignore[attr-defined]
    ViTPatchGenerator.__init__ = initialize


@contextmanager
def mlx_vlm_compatible_model_path(
    model_name: str | Path,
    *,
    hf_config: Any | None = None,
    revision: str | None = None,
) -> Iterator[Path]:
    """Let mlx-vlm load Nano VL V2 without changing checkpoint files.

    The public Nano checkpoint declares ``NemotronH_Nano_VL_V2`` while
    mlx-vlm v0.6.8 names its matching implementation
    ``nemotron_h_nano_omni``. Its loader imports a module by lowercasing
    ``model_type`` and has no alias for the former. A temporary directory
    links every checkpoint file but substitutes the model-type alias and
    disables Omni's separate video patch projection. The vLLM processor sees the original
    checkpoint config and architecture.
    """
    model_path = Path(model_name)
    if (
        not model_path.exists()
        and getattr(hf_config, "model_type", None) == _NANO_VL_ARCHITECTURE
        and _NANO_VL_ARCHITECTURE in (getattr(hf_config, "architectures", ()) or ())
    ):
        # Ordinary Hugging Face IDs have not been resolved by the lifecycle's
        # ModelScope helper. Use mlx-vlm's normal resolver, at the requested
        # revision, before making the same local config view.
        from mlx_vlm.utils import get_model_path

        model_path = get_model_path(str(model_name), revision=revision)
    config_path = model_path / "config.json"
    if not config_path.is_file():
        yield model_path
        return

    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get(
        "model_type"
    ) != _NANO_VL_ARCHITECTURE or _NANO_VL_ARCHITECTURE not in (
        config.get("architectures") or ()
    ):
        yield model_path
        return

    with TemporaryDirectory(prefix="vllm-metal-nano-vl-") as tmpdir:
        compatible_path = Path(tmpdir)
        for source in model_path.iterdir():
            if source.name == "config.json":
                continue
            (compatible_path / source.name).symlink_to(
                source.resolve(), target_is_directory=source.is_dir()
            )
        compatible_config = dict(config)
        compatible_config["model_type"] = _NANO_VL_MLX_VLM_MODEL_TYPE
        compatible_config["vision_config"] = {
            **config.get("vision_config", {}),
            "separate_video_embedder": False,
        }
        _respect_nano_image_only_vision_config()
        (compatible_path / "config.json").write_text(
            json.dumps(compatible_config), encoding="utf-8"
        )
        logger.info(
            "Using mlx-vlm Nano VL V2 model-type compatibility view for %s",
            model_path,
        )
        yield compatible_path
