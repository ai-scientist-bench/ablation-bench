import io
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score

from ablation_bench.judge.abstract import Judge, register_judge
from ablation_bench.types import (
    AblationSuggestionPred,
    DatasetForEvaluation,
    EvaluationResult,
    MajorityJudgeConfig,
    MissingAblationSuggestionPred,
    SingleResult,
)


@register_judge("majority_judge")
class MajorityJudge(Judge):
    """Judge that computes majority vote from multiple other judges."""

    def __init__(self, config: MajorityJudgeConfig) -> None:
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.judge_output_dirs = config.judge_output_dirs
        self.map_func = {
            DatasetForEvaluation.ResearcherAssist.value: self._evaluate_instance_researcherassist,
            DatasetForEvaluation.ReviewerAssist.value: self._evaluate_instance_reviewerassist,
        }

    @classmethod
    def from_config(
        cls, config_path: Path, model_name: str, reasoning_effort: str | None, output_dir: Path, parallelism: int = 1
    ) -> "Judge":
        """Create judge from config file."""
        config = yaml.safe_load(config_path.read_text())
        config["model"] = {"name": model_name, "reasoning_effort": reasoning_effort}
        config["output_dir"] = output_dir
        config["parallelism"] = parallelism
        config = MajorityJudgeConfig(**config)
        return cls(config=config)

    def _load_task_predictions(self, task_id: str) -> list[list[dict]]:
        """Load predictions for a task from all judge output directories."""
        all_predictions = []

        for judge_dir in self.judge_output_dirs:
            prediction_file = Path(judge_dir) / f"{task_id}.jsonl"
            if prediction_file.exists():
                try:
                    predictions = []
                    with prediction_file.open("r") as f:
                        for line in f:
                            if line.strip():
                                pred = json.loads(line.strip())
                                predictions.append(pred)
                    all_predictions.append(predictions)
                    self.logger.debug(f"Loaded {len(predictions)} predictions from {prediction_file}")
                except Exception as e:
                    self.logger.error(f"Failed to load predictions from {prediction_file}: {e}")
            else:
                self.logger.warning(f"Prediction file not found: {prediction_file}")

        return all_predictions

    def _load_task_cost(self, task_id: str) -> float:
        """Load total cost for a task from all judge evaluation files."""
        total_cost = 0.0

        for judge_dir in self.judge_output_dirs:
            evaluations_file = Path(judge_dir) / "evaluations.json"
            if evaluations_file.exists():
                try:
                    evaluations_df = pd.read_json(evaluations_file, orient="records")
                    task_row = evaluations_df[evaluations_df["id"] == task_id]
                    if not task_row.empty:
                        task_cost = task_row.iloc[0].get("cost", 0.0)
                        total_cost += task_cost
                        self.logger.debug(f"Added cost {task_cost} from {judge_dir} for task {task_id}")
                except Exception as e:
                    self.logger.error(f"Failed to load cost from {evaluations_file}: {e}")
            else:
                self.logger.warning(f"Evaluations file not found: {evaluations_file}")

        return total_cost

    def _compute_majority_predictions_researcherassist(
        self, all_predictions: list[list[dict]]
    ) -> list[AblationSuggestionPred]:
        """Compute majority vote for research assist predictions."""
        if not all_predictions:
            return []

        # Group predictions by name_in_paper
        paper_to_plan_votes = defaultdict(list)

        for judge_predictions in all_predictions:
            for pred in judge_predictions:
                name_in_paper = pred.get("name_in_paper")
                name_in_plan = pred.get("name_in_plan")
                if name_in_paper:
                    if isinstance(name_in_plan, list):
                        name_in_plan = frozenset(name_in_plan)  # Use frozenset for multiple plans
                    paper_to_plan_votes[name_in_paper].append(name_in_plan)

        # Compute majority for each name_in_paper
        majority_predictions = []
        for name_in_paper, plan_votes in paper_to_plan_votes.items():
            # Count votes (including None/null votes)
            vote_counts = Counter(plan_votes)
            most_common = vote_counts.most_common(1)

            if most_common:
                winning_vote, count = most_common[0]
                # Check if it's actually a majority (more than half)
                if count > len(plan_votes) // 2:
                    majority_plan = winning_vote
                else:
                    majority_plan = None
            else:
                majority_plan = None

            majority_predictions.append(AblationSuggestionPred(name_in_paper=name_in_paper, name_in_plan=majority_plan))

        return majority_predictions

    def _compute_majority_predictions_reviewerassist(
        self, all_predictions: list[list[dict]]
    ) -> list[MissingAblationSuggestionPred]:
        """Compute majority vote for reviewer assist predictions."""
        if not all_predictions:
            return []

        # Group predictions by name_in_plan
        plan_to_review_votes = defaultdict(list)

        for judge_predictions in all_predictions:
            for pred in judge_predictions:
                name_in_plan = pred.get("name_in_plan")
                appears_in_review = pred.get("appears_in_review", False)
                if name_in_plan:
                    plan_to_review_votes[name_in_plan].append(appears_in_review)

        # Compute majority for each name_in_plan
        majority_predictions = []
        for name_in_plan, review_votes in plan_to_review_votes.items():
            # Count votes
            true_votes = sum(review_votes)
            total_votes = len(review_votes)

            # Majority vote for appears_in_review
            appears_majority = true_votes > total_votes // 2

            majority_predictions.append(
                MissingAblationSuggestionPred(name_in_plan=name_in_plan, appears_in_review=appears_majority)
            )

        return majority_predictions

    def _evaluate_instance_researcherassist(
        self,
        task: dict[str, Any],
        predictions_path: Path,
        top_k: int | None,
    ) -> dict[str, Any]:
        """Evaluate a single research assist task using majority vote."""
        task_id = task["id"]

        # Load predictions from all judge output directories
        all_predictions = self._load_task_predictions(task_id)

        if not all_predictions:
            self.logger.warning(f"No predictions found for task {task_id}")
            return {
                "id": task_id,
                "true_labels": [],
                "pred_labels": [],
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "cost": 0.0,
            }

        # Compute majority predictions
        majority_preds = self._compute_majority_predictions_researcherassist(all_predictions)

        # Save majority predictions to output directory
        output_file = self.config.output_dir / f"{task_id}.jsonl"
        with output_file.open("w") as f:
            for pred in majority_preds:
                f.write(pred.model_dump_json() + "\n")

        # Load the original data for evaluation (similar to SimpleLMJudge)
        ablations_in_plan = pd.read_json(predictions_path / f"{task_id}.jsonl", lines=True)
        if top_k:
            ablations_in_plan = ablations_in_plan.head(top_k)

        ablations_in_paper = pd.read_json(io.StringIO(task["ablations_in_paper"]))

        # Calculate metrics
        true_labels, pred_labels = self._get_labels(
            predictions=majority_preds,
            ablations_in_paper=[
                ablation.get("name")
                for ablation in ablations_in_paper.replace({float("nan"): None}).to_dict(orient="records")
            ],
            ablations_in_plan=[
                ablation.get("name")
                for ablation in ablations_in_plan.replace({float("nan"): None}).to_dict(orient="records")
            ],
        )

        # Calculate metrics
        if len(pred_labels) > 0 and len(true_labels) > 0:
            precision = precision_score(true_labels, pred_labels) if any(pred_labels) else 0.0
            recall = recall_score(true_labels, pred_labels) if any(true_labels) else 0.0
            f1 = f1_score(true_labels, pred_labels) if any(true_labels) or any(pred_labels) else 0.0
            ndcg_score = self._ndcg_score(
                true_labels, pred_labels, k=min(len(ablations_in_plan), len(ablations_in_paper))
            )
        else:
            precision = recall = f1 = 0.0

        # Load total cost from all judges for this task
        total_cost = self._load_task_cost(task_id)

        return {
            "id": task_id,
            "true_labels": true_labels,
            "pred_labels": pred_labels,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "ndcg_score": ndcg_score,
            "cost": total_cost,
        }

    def _evaluate_instance_reviewerassist(
        self,
        task: dict[str, Any],
        predictions_path: Path,
        top_k: int | None,
    ) -> dict[str, Any]:
        """Evaluate a single reviewer assist task using majority vote."""
        task_id = task["id"]

        # Load predictions from all judge output directories
        all_predictions = self._load_task_predictions(task_id)

        if not all_predictions:
            self.logger.warning(f"No predictions found for task {task_id}")
            return {
                "id": task_id,
                "true_labels": [],
                "pred_labels": [],
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "cost": 0.0,
            }

        # Compute majority predictions
        majority_preds = self._compute_majority_predictions_reviewerassist(all_predictions)

        # Apply top_k if specified
        if top_k:
            majority_preds = majority_preds[:top_k]

        # Save majority predictions to output directory
        output_file = self.config.output_dir / f"{task_id}.jsonl"
        with output_file.open("w") as f:
            for pred in majority_preds:
                f.write(pred.model_dump_json() + "\n")

        # Calculate metrics (similar to SimpleLMJudge logic)
        num_ablations = task["num_ablation_suggestions"]
        overlap_ablations = min(sum([pred.appears_in_review for pred in majority_preds]), num_ablations)
        non_overlap_ablations = max(0, num_ablations - overlap_ablations)
        non_existing_ablations = sum([not pred.appears_in_review for pred in majority_preds])

        true_labels, pred_labels = (
            [True] * num_ablations + [False] * non_existing_ablations,
            [True] * overlap_ablations + [False] * non_overlap_ablations + [True] * non_existing_ablations,
        )

        # Calculate metrics
        if len(pred_labels) > 0 and len(true_labels) > 0:
            precision = precision_score(true_labels, pred_labels) if any(pred_labels) else 0.0
            recall = recall_score(true_labels, pred_labels) if any(true_labels) else 0.0
            f1 = f1_score(true_labels, pred_labels) if any(true_labels) or any(pred_labels) else 0.0
        else:
            precision = recall = f1 = 0.0

        # Load total cost from all judges for this task
        total_cost = self._load_task_cost(task_id)

        return {
            "id": task_id,
            "true_labels": true_labels,
            "pred_labels": pred_labels,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cost": total_cost,
        }

    def evaluate(self, predictions_path: Path, dataset: Dataset, top_k: int | None) -> EvaluationResult:
        """Evaluate using majority vote from multiple judges."""
        dataset_full_name = f"talor-abramovich/{dataset.info.dataset_name}"

        with_labels = dataset.map(
            lambda task: self.map_func[dataset_full_name](task, predictions_path, top_k),
            remove_columns=[column for column in dataset.column_names if column != "id"],
            desc="Evaluating instances using majority judge",
            num_proc=self.config.parallelism,
            load_from_cache_file=False,
        )

        with_labels_df = with_labels.to_pandas()
        with_labels_df.to_json(self.config.output_dir / "evaluations.json", orient="records", indent=4)

        self.logger.info(f"Processed {len(with_labels_df)} tasks with majority voting")

        # Calculate overall metrics
        scores_to_return = (
            ["precision", "recall", "f1_score"]
            if dataset_full_name == DatasetForEvaluation.ReviewerAssist.value
            else [
                "precision",
                "recall",
                "f1_score",
                "ndcg_score",
            ]
        )

        result = with_labels_df[[*scores_to_return, "cost"]].mean()
        result_std_dev = with_labels_df[scores_to_return].std()

        return EvaluationResult(
            precision=SingleResult(result=result.precision, std_dev=result_std_dev.precision),
            recall=SingleResult(result=result.recall, std_dev=result_std_dev.recall),
            f1_score=SingleResult(result=result.f1_score, std_dev=result_std_dev.f1_score),
            ndcg_score=SingleResult(result=result.ndcg_score, std_dev=result_std_dev.ndcg_score)
            if "ndcg_score" in scores_to_return
            else None,
            cost=result.cost,
        )
