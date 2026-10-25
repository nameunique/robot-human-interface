"""Explicit, CWD-independent model selection for application entry points.

The GUI and CLI use Volodka. Low-level simulation/retargeting constructors
retain their legacy defaults for existing integrations; callers select a
profile here when composing a complete pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from robot_human_interface.resources import ResourceLocator


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    label_ru: str
    model_directory: str
    joint_config_name: str
    balance_config_name: str
    hardware_mapping_verified: bool
    hardware_block_reason: str = ""

    def scene_path(self, mode: str, resources: ResourceLocator | None = None) -> Path:
        if mode not in {"fixed", "free"}:
            raise ValueError("scene mode must be fixed or free")
        return (resources or ResourceLocator()).model(
            self.model_directory, f"scene_{mode}.xml"
        )

    def joints_path(self, resources: ResourceLocator | None = None) -> Path:
        return (resources or ResourceLocator()).config(self.joint_config_name)

    def balance_path(self, resources: ResourceLocator | None = None) -> Path:
        return (resources or ResourceLocator()).config(self.balance_config_name)


DEFAULT_MODEL_PROFILE = "volodka"
MODEL_PROFILES = MappingProxyType({
    "volodka": ModelProfile(
        "volodka", "Volodka · CAD-модель", "volodka",
        "joints_volodka.yaml", "balance_volodka.yaml", False,
        "Отправка на робота заблокирована: соответствие углов CAD-модели "
        "реальным приводам ещё не проверено на аппаратном стенде.",
    ),
    "unity_legacy": ModelProfile(
        "unity_legacy", "Unity · прежняя модель", "humanoid",
        "joints.yaml", "balance.yaml", True,
    ),
})


def get_model_profile(name: str = DEFAULT_MODEL_PROFILE) -> ModelProfile:
    """Reject unknown profiles rather than silently substituting a model."""
    try:
        return MODEL_PROFILES[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unknown model profile: {name!r}") from error


__all__ = ["DEFAULT_MODEL_PROFILE", "MODEL_PROFILES", "ModelProfile", "get_model_profile"]
