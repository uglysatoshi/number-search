"""Explicit, validated model configuration and path discovery."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ModelError


def default_model_dir():
    configured = os.environ.get("NUMBER_SEARCH_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file():
        return root / "models" / "v5"
    return Path.cwd() / "models" / "v5"


@dataclass(frozen=True)
class ModelConfig:
    schema_version: int = 1
    detector_size: tuple[int, int] = (512, 384)
    temperatures: tuple[float, float] = (1.3, 1.3)
    glyph_fusion_weight: float = 0.25
    recovery_log_score: float = -0.3
    tile_recovery: bool = True

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ModelError("pipeline.json has missing or unsupported settings")
        try:
            config = cls(
                **{
                    **value,
                    "detector_size": tuple(value["detector_size"]),
                    "temperatures": tuple(value["temperatures"]),
                }
            )
            valid = (
                type(config.schema_version) is int
                and config.schema_version == 1
                and len(config.detector_size) == 2
                and all(type(v) is int and 32 <= v <= 2048 for v in config.detector_size)
                and len(config.temperatures) == 2
                and all(
                    type(v) in (int, float) and math.isfinite(v) and v > 0
                    for v in config.temperatures
                )
                and type(config.glyph_fusion_weight) in (int, float)
                and math.isfinite(config.glyph_fusion_weight)
                and 0 <= config.glyph_fusion_weight <= 1
                and type(config.recovery_log_score) in (int, float)
                and math.isfinite(config.recovery_log_score)
                and config.recovery_log_score <= 0
                and type(config.tile_recovery) is bool
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ModelError(f"Invalid pipeline settings: {exc}") from exc
        if not valid:
            raise ModelError("Invalid values in pipeline.json")
        return config
