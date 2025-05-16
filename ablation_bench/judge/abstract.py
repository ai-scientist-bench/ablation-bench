"""Judge module for ablations-bench."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import Dataset

from ablation_bench.types import AblationSuggestionPred, EvaluationResult

# Registry to store judge classes
JUDGE_REGISTRY: dict[str, type["Judge"]] = {}


def register_judge(name: str):
    """Decorator to register a judge class."""

    def decorator(cls):
        JUDGE_REGISTRY[name.lower()] = cls
        return cls

    return decorator


class Judge(ABC):
    """Abstract base class for ablation judges."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the judge.

        Args:
            config: A dictionary containing the judge-specific configuration.
        """
        self.config = config

    @classmethod
    @abstractmethod
    def from_config(cls, config_path: Path, model_name: str, output_dir: Path, parallelism: int = 1) -> "Judge":
        """Create a judge instance from a config file.

        Args:
            config_path: Path to the configuration file.
            model_name: Name of the model being judged.
            output_dir: Directory to save any outputs.
            parallelism: Number of parallel processes to use, if applicable.

        Returns:
            An instance of the Judge.
        """
        raise NotImplementedError

    @staticmethod
    def _get_labels(
        predictions: list[AblationSuggestionPred],
        ablations_in_paper: list[str],
        ablations_in_plan: list[str],
    ) -> tuple[list[bool], list[bool]]:
        """Get labels for the predictions.

        Args:
            predictions: List of predictions
            ablations_in_paper: List of ablations in the paper
            ablations_in_plan: List of ablations in the plan

        Returns:
            true_labels, pred_labels Tuple of list of boolean labels
        """
        true_labels = [True] * len(ablations_in_paper)
        pred_labels = []

        # Process paper ablations
        for ablation in ablations_in_paper:
            pred_labels.append(
                any(
                    pred.name_in_paper == ablation and pred.label and pred.name_in_plan in ablations_in_plan
                    for pred in predictions
                )
            )

        # Process additional plan ablations (false positives)
        additional_plan_ablations = len(
            list(
                set(ablation for ablation in ablations_in_plan)
                - set(pred.name_in_plan for pred in predictions if pred.name_in_plan is not None)
            )
        )
        true_labels.extend([False] * additional_plan_ablations)
        pred_labels.extend([True] * additional_plan_ablations)
        return true_labels, pred_labels

    @abstractmethod
    def evaluate(self, predictions_path: Path, dataset: Dataset, top_k: int | None) -> EvaluationResult:
        """Evaluate predictions against ground truth.

        Args:
            predictions_path: Path to the predictions file.
            dataset: Dataset to use for evaluation.
            top_k: Consider only the top K predictions, if applicable.

        Returns:
            An EvaluationResult object containing precision, recall, F1-score, and cost.
        """
        raise NotImplementedError

    @staticmethod
    def get_judge(name: str) -> type["Judge"]:
        """Get a judge class by name.

        Args:
            name: Name of the judge class

        Returns:
            Judge class corresponding to the name

        Raises:
            ValueError: If the judge class is not found
        """
        judge_class = JUDGE_REGISTRY.get(name.lower())
        if judge_class is None:
            raise ValueError(f"Judge '{name}' not found. Available judges: {list(JUDGE_REGISTRY.keys())}")
        return judge_class
