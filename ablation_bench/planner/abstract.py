"""Abstract base classes and registry for ablation planners."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import Dataset

# Registry to store planner classes
PLANNER_REGISTRY: dict[str, type["Planner"]] = {}


def register_planner(name: str):
    """Decorator to register a planner class."""

    def decorator(cls):
        PLANNER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


class Planner(ABC):
    """Abstract base class for ablation planners."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the planner.

        Args:
            config: A dictionary containing the planner-specific configuration.
        """
        self.config = config

    @classmethod
    @abstractmethod
    def from_config(
        cls, config_path: Path, model_name: str, output_dir: Path, num_ablations: int = 5, parallelism: int = 1
    ) -> "Planner":
        """Create a planner instance from a config file."""
        raise NotImplementedError

    @abstractmethod
    def plan(self, dataset: Dataset) -> None:
        """Plan ablations for the given dataset."""
        raise NotImplementedError

    @staticmethod
    def get_planner(name: str) -> type["Planner"]:
        """Get a planner class by name."""
        planner_class = PLANNER_REGISTRY.get(name.lower())
        if planner_class is None:
            raise ValueError(f"Planner '{name}' not found. Available planners: {list(PLANNER_REGISTRY.keys())}")
        return planner_class
