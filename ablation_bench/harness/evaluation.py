"""Evaluation module for ablations-bench."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, get_args

import typer
from enum import Enum
from datasets import Dataset, load_dataset
from pydantic import Field
from pydantic_settings import BaseSettings

from ablation_bench.harness.utils import get_field_description
from ablation_bench.judge import Judge, JudgeType
from ablation_bench.logger import get_logger
from ablation_bench.types import DatasetForEvaluation, DatasetSplit, EvaluationResult

app = typer.Typer(name="eval", help="Run ablation evaluation", no_args_is_help=True)


class EvaluationSettings(BaseSettings):
    """Settings for evaluation run."""

    judge: JudgeType = Field(description="Name of the judge class to use for evaluation.")  # type: ignore
    judge_config: Path = Field(description="Path to the judge configuration file (YAML format)")
    model_name: str = Field(
        description="Name of the model used to evaluate (e.g., 'openai/gpt-4o'). Any LiteLLM supported model is valid."
    )
    dataset: DatasetForEvaluation = Field(
        description="Name of the dataset in HuggingFace hub (e.g., 'tau/ablations-bench')"
    )
    split: DatasetSplit = Field(default="dev", description="Dataset split to use for evaluation (e.g., 'dev', 'test')")
    generated_plans_path: Path = Field(description="Path to the directory containing generated ablation plans")
    top_k: int | None = Field(
        default=None,
        description="Number of k generated plans to calculate metrics@k. "
        "If None, all the generated plans will be evaluated",
    )
    output_dir: Path | None = Field(default=None, description="Path to the directory to save evaluation results")
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default=None, description="Reasoning effort level for the judge model"
    )
    parallelism: int = Field(default=1, description="Number of parallel workers to use for evaluation")

    class Config:
        env_prefix = "ABLATIONS_"


class Evaluator:
    """Main evaluator class for running model evaluations."""

    def __init__(self, settings: EvaluationSettings):
        """Initialize the Evaluator with evaluation settings.

        Args:
            settings: The EvaluationSettings object containing all configurations.
        """
        self.settings = settings
        self.dataset: Dataset | None = None
        self.judge: Judge | None = None
        self.logger = get_logger(__name__)

        # Create output directory
        if settings.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            settings.output_dir = (Path("runs") / timestamp).absolute()

        settings.output_dir.mkdir(parents=True, exist_ok=True)

    def load_judge(self) -> Judge:
        """Load judge class and instantiate with config."""
        self.logger.info(
            f"Loading judge {self.settings.judge.value} and instantiating with config {self.settings.judge_config}"
        )
        judge_class = Judge.get_judge(self.settings.judge.value)
        self.judge = judge_class.from_config(
            self.settings.judge_config,
            model_name=self.settings.model_name,
            parallelism=self.settings.parallelism,
            output_dir=self.settings.output_dir,
            reasoning_effort=self.settings.reasoning_effort,
        )
        return self.judge

    def load_data(self) -> Dataset:
        """Load dataset from Hugging Face."""
        self.logger.info(f"Loading dataset {self.settings.dataset.value}, split {self.settings.split.value}")
        self.dataset = load_dataset(self.settings.dataset.value, split=self.settings.split.value)
        return self.dataset

    def run_evaluation(self) -> EvaluationResult:
        """Run the complete evaluation process."""
        # Load dataset
        self.load_data()

        # Initialize judge
        self.load_judge()

        # Run evaluation
        results = self.judge.evaluate(
            dataset=self.dataset, predictions_path=self.settings.generated_plans_path, top_k=self.settings.top_k
        )
        
        ndcg_info = (
            f"NDCG: {results.ndcg_score.result:.2f} ± {results.ndcg_score.std_dev:.2f}, "
            if results.ndcg_score
            else ""
        )

        self.logger.info(
            f"Evaluation completed. "
            f"Precision: {results.precision.result:.2f} ± {results.precision.std_dev:.2f}, "
            f"Recall: {results.recall.result:.2f} ± {results.recall.std_dev:.2f} ,"
            f"F1: {results.f1_score.result:.2f} ± {results.f1_score.std_dev:.2f}, "
            f"{ndcg_info}"
            f"Cost: {results.cost:.2f}"
        )

        return results


@app.callback(invoke_without_command=True)
def evaluate(
    judge: Annotated[JudgeType, typer.Option(help=get_field_description(EvaluationSettings, "judge"))],
    judge_config: Annotated[Path, typer.Option(help=get_field_description(EvaluationSettings, "judge_config"))],
    model_name: Annotated[str, typer.Option(help=get_field_description(EvaluationSettings, "model_name"))],
    dataset: Annotated[DatasetForEvaluation, typer.Option(help=get_field_description(EvaluationSettings, "dataset"))],
    split: Annotated[DatasetSplit, typer.Option(help=get_field_description(EvaluationSettings, "split"))],
    generated_plans_path: Annotated[
        Path, typer.Option(help=get_field_description(EvaluationSettings, "generated_plans_path"))
    ],
    top_k: Annotated[int | None, typer.Option(help=get_field_description(EvaluationSettings, "top_k"))] = None,
    parallelism: Annotated[int, typer.Option(help=get_field_description(EvaluationSettings, "parallelism"))] = 1,
    output_dir: Annotated[
        Path | None, typer.Option(help=get_field_description(EvaluationSettings, "output_dir"))
    ] = None,
    reasoning_effort: Annotated[
        Enum("ReasoningEffort", {key: key for key in get_args(Literal["low", "medium", "high"])}) | None,
        typer.Option(help=get_field_description(EvaluationSettings, "reasoning_effort")),
    ] = None,
) -> None:
    """Run ablation evaluation with the specified settings."""
    settings = EvaluationSettings(
        judge=judge,
        judge_config=judge_config,
        dataset=dataset,
        split=split,
        generated_plans_path=generated_plans_path,
        model_name=model_name,
        top_k=top_k,
        parallelism=parallelism,
        output_dir=output_dir,
        reasoning_effort=reasoning_effort.value if reasoning_effort is not None else None,
    )
    evaluator = Evaluator(settings)
    evaluator.run_evaluation()
