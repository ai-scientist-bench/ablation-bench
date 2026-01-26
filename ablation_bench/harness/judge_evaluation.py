"""Judge evaluation module for ablations-bench."""

import io
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from datasets import Dataset, load_dataset
from pydantic import Field
from pydantic_settings import BaseSettings
from sklearn.metrics import f1_score, precision_score, recall_score

from ablation_bench.harness.utils import get_field_description
from ablation_bench.logger import get_logger
from ablation_bench.types import (
    DatasetForJudgeEvaluation,
    EvaluationResult,
    NonPredictedField,
    PredictedField,
    SingleResult,
)

app = typer.Typer(name="eval-judge", help="Run judge evaluation", no_args_is_help=True)


class JudgeEvaluationSettings(BaseSettings):
    """Settings for judge evaluation run."""

    dataset: DatasetForJudgeEvaluation = Field(description="Path to dataset for judge evaluation (e.g., 'data/ai-coscientist/reviewer-eval')")
    judge_evaluations_path: Path = Field(description="Path to the directory containing judge evaluation results")

    class Config:
        env_prefix = "ABLATIONS_"


class JudgeEvaluator:
    """Main evaluator class for running judge evaluations."""

    def __init__(self, settings: JudgeEvaluationSettings):
        """Initialize the JudgeEvaluator with evaluation settings.

        Args:
            settings: The JudgeEvaluationSettings object containing all configurations.
        """
        self.settings = settings
        self.dataset: Dataset | None = None
        self.logger = get_logger(__name__)

    def load_data(self) -> Dataset:
        """Load dataset from local parquet files."""
        self.logger.info(f"Loading dataset {self.settings.dataset.value}")
        self.dataset = load_dataset(
            "parquet", 
            data_files={"test": f"{self.settings.dataset.value}/test.parquet"}, 
            split="test"
        )
        self.dataset.info.dataset_name = self.settings.dataset.value
        return self.dataset

    def load_ground_truth(self, evaluation_gt: str) -> pd.DataFrame:
        """Load ground truth labels from evaluation_gt string."""
        gt = pd.read_json(io.StringIO(evaluation_gt), lines=True)
        gt["gt_label"] = gt[PredictedField[self.settings.dataset]].apply(bool)
        gt.drop(columns=[PredictedField[self.settings.dataset]], inplace=True)
        return gt

    def load_predictions(self, instance_id: str) -> pd.DataFrame | None:
        """Load predictions from judge evaluation file."""
        eval_file = self.settings.judge_evaluations_path / f"{instance_id}.jsonl"
        if not eval_file.exists():
            return None

        predictions = pd.read_json(io.StringIO(eval_file.read_text()), lines=True)
        predictions["pred_label"] = predictions[PredictedField[self.settings.dataset]].apply(bool)
        predictions.drop(columns=[PredictedField[self.settings.dataset]], inplace=True)
        return predictions

    def load_instance_cost(self, instance_id: str) -> float:
        """Load cost for a specific instance."""
        model_name, task_id = instance_id.split("/")
        cost_file = self.settings.judge_evaluations_path / model_name / "evaluations.json"
        if not cost_file.exists():
            self.logger.warning(f"Cost evaluations file does not exist: {cost_file}")
            return 0.0

        costs = pd.read_json(io.StringIO(cost_file.read_text()))
        try:
            return costs.loc[costs["id"] == task_id]["cost"].values[0]
        except IndexError:
            self.logger.warning(f"Task ID {task_id} not found in cost file {cost_file} or cost not available.")
            return 0.0
        except Exception as e:
            self.logger.error(f"Error reading cost for task ID {task_id} from {cost_file}: {e}")
            return 0.0

    def evaluate_instance(self, instance: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a single instance."""
        gt_df = self.load_ground_truth(instance["labels"])
        pred_df = self.load_predictions(instance["id"])
        cost = self.load_instance_cost(instance["id"])

        result = EvaluationResult(
            precision=SingleResult(result=0.0),
            recall=SingleResult(result=0.0),
            f1_score=SingleResult(result=0.0),
            cost=cost,
        )
        if pred_df is not None:
            merged_df = gt_df.merge(pred_df, on=NonPredictedField[self.settings.dataset], how="left")
            gt_labels = merged_df["gt_label"].tolist()
            pred_labels = merged_df["pred_label"].replace({float("nan"): False}).tolist()
            result = EvaluationResult(
                precision=SingleResult(result=precision_score(gt_labels, pred_labels, zero_division=0)),
                recall=SingleResult(result=recall_score(gt_labels, pred_labels, zero_division=0)),
                f1_score=SingleResult(result=f1_score(gt_labels, pred_labels, zero_division=0)),
                cost=cost,
            )
        instance["precision"] = result.precision.result
        instance["recall"] = result.recall.result
        instance["f1_score"] = result.f1_score.result
        instance["cost"] = result.cost
        return instance

    def run_evaluation(self) -> EvaluationResult:
        """Run the complete evaluation process."""
        self.load_data()

        with_labels = self.dataset.map(
            self.evaluate_instance, desc="Evaluating judge", load_from_cache_file=False
        ).to_pandas()

        result = with_labels[["precision", "recall", "f1_score", "cost"]].mean()
        result_std_dev = with_labels[["precision", "recall", "f1_score"]].std()

        # Calculate macro averages
        return EvaluationResult(
            precision=SingleResult(result=result.precision, std_dev=result_std_dev.precision),
            recall=SingleResult(result=result.recall, std_dev=result_std_dev.recall),
            f1_score=SingleResult(result=result.f1_score, std_dev=result_std_dev.f1_score),
            cost=result.cost,
        )


@app.callback(invoke_without_command=True)
def evaluate_judge(
    dataset: Annotated[
        DatasetForJudgeEvaluation, typer.Option(help=get_field_description(JudgeEvaluationSettings, "dataset"))
    ],
    judge_evaluations_path: Annotated[
        Path, typer.Option(help=get_field_description(JudgeEvaluationSettings, "judge_evaluations_path"))
    ],
) -> None:
    """Run judge evaluation with the specified settings."""
    settings = JudgeEvaluationSettings(
        dataset=dataset,
        judge_evaluations_path=judge_evaluations_path,
    )

    evaluator = JudgeEvaluator(settings)
    eval_result = evaluator.run_evaluation()
    evaluator.logger.info(
        f"Evaluation completed. "
        f"Precision: {eval_result.precision.result:.2f} ± {eval_result.precision.std_dev:.2f}, "
        f"Recall: {eval_result.recall.result:.2f} ± {eval_result.recall.std_dev:.2f} ,"
        f"F1: {eval_result.f1_score.result:.2f} ± {eval_result.f1_score.std_dev:.2f}, "
        f"Cost: {eval_result.cost:.2f}"
    )
