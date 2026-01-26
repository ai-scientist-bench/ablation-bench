import io
import json
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import Dataset
from jinja2 import Template
from litellm import ModelResponse, completion, completion_cost
from sklearn.metrics import f1_score, precision_score, recall_score
from tenacity import retry, stop_after_attempt, wait_exponential

from ablation_bench.judge.abstract import Judge, register_judge
from ablation_bench.logger import get_logger
from ablation_bench.types import (
    AblationSuggestionPred,
    AblationSuggestionPredResponse,
    DatasetForEvaluation,
    EvaluationResult,
    MissingAblationSuggestionPred,
    MissingAblationSuggestionPredResponse,
    PredResponse,
    SimpleLMConfig,
    SingleResult,
)

TEXT_FILE_EXT = ["*.tex", "*.text", ".txt", "*.bib", "*.bbl", "*.md"]


@register_judge("simple_lm")
class SimpleLMJudge(Judge):
    """Judge that uses a language model to evaluate ablations."""

    def __init__(self, config: SimpleLMConfig) -> None:
        """Initialize the SimpleLMJudge.

        Args:
            config: The configuration object for the judge.
        """
        super().__init__(config)
        self.system_prompt = config.prompts["system"]
        self.user_prompt = config.prompts["user"]
        self.config = config
        self.logger = get_logger(__name__)
        self.map_func = {
            DatasetForEvaluation.ResearcherAssist.value: self._evaluate_instance_researcherassist,
            DatasetForEvaluation.ReviewerAssist.value: self._evaluate_instance_reviewerassist,
        }
        self.map_type = {
            DatasetForEvaluation.ResearcherAssist.value: AblationSuggestionPredResponse,
            DatasetForEvaluation.ReviewerAssist.value: MissingAblationSuggestionPredResponse,
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
        config = SimpleLMConfig(**config)
        return cls(config=config)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=lambda retry_state: logging.warning(
            f"Retry attempt {retry_state.attempt_number} after error: {retry_state.outcome.exception()}"
        ),
    )
    def _get_lm_response(
        self,
        task: dict[str, Any],
        return_class: type["PredResponse"],
        **kwargs: dict[str, Any],
    ) -> PredResponse:
        """Get LM response for a single task."""
        task_id = task["id"]

        # First try to get it from the log file before making a request to LM
        log_path = self.config.output_dir / f"{task_id}.json.log"
        if log_path.is_file():
            try:
                response = ModelResponse(**json.loads(log_path.read_text()))
                cost = completion_cost(response)
                return return_class.from_lm_response(
                    response.choices[0].message.content,
                    cost=cost,
                    post_process_prediction=kwargs.get("post_process_predictions", None),
                )
            except Exception as ex:
                self.logger.warning(f"Failed to read log file {log_path}: {ex}")

        user_prompt_template = Template(self.user_prompt)
        formatted_user_prompt = user_prompt_template.render(**kwargs)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": formatted_user_prompt},
        ]

        completion_kwargs = {}
        if self.config.model.reasoning_effort is not None:
            completion_kwargs["reasoning_effort"] = self.config.model.reasoning_effort

        response = completion(
            model=self.config.model.name,
            messages=messages,
            temperature=self.config.model.temperature,
            top_p=self.config.model.top_p,
            drop_params=True,
            **completion_kwargs,
        )
        cost = completion_cost(response)
        log_path.write_text(json.dumps(response.to_dict(mode="json"), indent=4))
        response_content = response.choices[0].message.content
        return return_class.from_lm_response(
            response_content, cost=cost, post_process_prediction=kwargs.get("post_process_predictions", None)
        )

    def _get_paper_source(self, paper_path: Path) -> str:
        """Get the source of the paper without ablations."""

        paper_source = ""
        for extension in TEXT_FILE_EXT:
            for file_path in paper_path.rglob(extension):
                paper_source += f'<file name="{file_path.relative_to(paper_path)}">\n'
                paper_source += file_path.read_text(encoding="utf-8")
                paper_source += "\n</file>\n"

        return paper_source

    def _process_task_researchassist(
        self,
        task: dict,
        plan: str,
        return_class: type["PredResponse"],
    ) -> tuple[list[AblationSuggestionPred], float]:
        """Process a single task and return true/pred labels."""

        ablations_in_paper = [json.dumps(j) for j in json.loads(task["ablations_in_paper"])]
        random.shuffle(ablations_in_paper)
        plan = [p for p in plan.split("\n") if p.strip() != ""]
        random.shuffle(plan)
        ablations_in_paper = "\n".join(ablations_in_paper)
        plan = "\n".join(plan)
        sides = [ablations_in_paper, plan]
        random.shuffle(sides)

        def post_process_predictions(preds: list[str]) -> list[str]:
            name_in_paper_side = "A" if sides[0] == ablations_in_paper else "B"
            name_in_plan_side = "B" if name_in_paper_side == "A" else "A"
            new_preds = []
            for pred in preds:
                pred = json.loads(pred)
                if pred[f"name_in_{name_in_paper_side}"] is None:
                    continue
                if isinstance(pred[f"name_in_{name_in_paper_side}"], list):
                    for name in pred[f"name_in_{name_in_paper_side}"]:
                        new_preds.append(
                            json.dumps(
                                {
                                    "name_in_paper": name,
                                    "name_in_plan": pred[f"name_in_{name_in_plan_side}"],
                                }
                            )
                        )
                else:
                    new_preds.append(
                        json.dumps(
                            {
                                "name_in_paper": pred[f"name_in_{name_in_paper_side}"],
                                "name_in_plan": pred[f"name_in_{name_in_plan_side}"],
                            }
                        )
                    )
            return new_preds

        ablation_preds = self._get_lm_response(
            task,
            return_class,
            paper_title=task["paper_title"],
            abstract=task["paper_abstract"],
            side_A=sides[0],
            side_B=sides[1],
            post_process_predictions=post_process_predictions,
            paper_source=self._get_paper_source(Path(task["paper_path"])),
        )

        return ablation_preds.predictions, ablation_preds.cost

    def _process_task_reviewerassist(
        self,
        task: dict,
        plan: str,
        return_class: type["PredResponse"],
    ) -> tuple[list[MissingAblationSuggestionPred], float]:
        """Process a single task for ReviewerAssist mode and return predictions and cost."""

        reviews = json.loads(task["review_text"])
        random.shuffle(reviews)
        official_reviews = "\n</official_review>\n\n\n<official_review>\n".join(reviews)
        if plan == "":
            self.logger.warning(f"Empty plan for task {task.get('id', 'UnknownID')}")
            return [], 0.0
        plan = [p for p in plan.split("\n") if p.strip() != ""]
        random.shuffle(plan)
        plan = "\n".join(plan)
        ablation_preds = self._get_lm_response(
            task,
            return_class,
            problem_statement=plan,
            paper_title=task["paper_title"],
            abstract=task["paper_abstract"],
            official_reviews=official_reviews,
            paper_source=self._get_paper_source(Path(task["paper_path"])),
        )

        return ablation_preds.predictions, ablation_preds.cost

    def _evaluate_instance_researcherassist(
        self,
        task: dict[str, Any],
        predictions_path: Path,
        top_k: int | None,
        return_class: type["PredResponse"],
    ) -> dict[str, Any]:
        """Evaluate a single instance for the ResearcherAssist benchmark.

        This method processes a task, retrieves LM predictions for ablation suggestions,
        compares them against ground truth ablations from the paper and the plan,
        and calculates precision, recall, and F1 score.

        Args:
            task: The task dictionary containing paper details and ground truth ablations.
            predictions_path: Path to the directory containing the predicted ablation plans.
            top_k: If specified, consider only the top K predictions from the plan.
            return_class: The Pydantic model type to parse the LM response into.

        Returns:
            The task dictionary updated with true labels, predicted labels, and scores.
        """
        task_id = task["id"]
        plan = (predictions_path / f"{task_id}.jsonl").read_text()
        preds, cost = self._process_task_researchassist(task, plan, return_class)
        (self.config.output_dir / f"{task_id}.jsonl").write_text("\n".join(pred.model_dump_json() for pred in preds))

        ablations_in_plan = pd.read_json(predictions_path / f"{task_id}.jsonl", lines=True)
        if top_k:
            ablations_in_plan = ablations_in_plan.head(top_k)

        ablations_in_paper = pd.read_json(io.StringIO(task["ablations_in_paper"]))
        # Calculate metrics
        true_labels, pred_labels = self._get_labels(
            predictions=preds,
            ablations_in_paper=[
                ablation.get("name")
                for ablation in ablations_in_paper.replace({float("nan"): None}).to_dict(orient="records")
            ],
            ablations_in_plan=[
                ablation.get("name")
                for ablation in ablations_in_plan.replace({float("nan"): None}).to_dict(orient="records")
            ],
        )
        task["true_labels"] = true_labels
        task["pred_labels"] = pred_labels
        task["precision"] = precision_score(true_labels, pred_labels)
        task["recall"] = recall_score(true_labels, pred_labels)
        task["f1_score"] = f1_score(true_labels, pred_labels)
        task["ndcg_score"] = self._ndcg_score(
            true_labels, pred_labels, k=min(len(ablations_in_plan), len(ablations_in_paper))
        )
        task["cost"] = cost
        return task

    def _evaluate_instance_reviewerassist(
        self,
        task: dict[str, Any],
        prediction_path: Path,
        top_k: int | None,
        return_class: type["PredResponse"],
    ) -> dict[str, Any]:
        """Evaluate a single instance for the ReviewerAssist benchmark.

        This method processes a task, retrieves LM predictions about whether ablations
        in a plan appear in reviews, compares them against the actual number of
        suggestions and overlaps, and calculates precision, recall, and F1 score.

        Args:
            task: The task dictionary containing review details and plan information.
            prediction_path: Path to the directory containing the predicted ablation plans.
            top_k: If specified, consider only the top K predictions from the plan.
            return_class: The Pydantic model type to parse the LM response into.

        Returns:
            The task dictionary updated with true labels, predicted labels, and scores.
        """
        task_id = task["id"]
        plan = (prediction_path / f"{task_id}.jsonl").read_text()
        preds, cost = self._process_task_reviewerassist(task, plan, return_class)
        (self.config.output_dir / f"{task_id}.jsonl").write_text("\n".join(pred.model_dump_json() for pred in preds))

        if top_k:
            preds = preds[:top_k]

        num_ablations = task["num_ablation_suggestions"]
        overlap_ablations = min(sum([pred.appears_in_review for pred in preds]), num_ablations)
        non_overlap_ablations = max(0, num_ablations - overlap_ablations)
        non_existing_ablations = sum([not pred.appears_in_review for pred in preds])

        true_labels, pred_labels = (
            [True] * num_ablations + [False] * non_existing_ablations,
            [True] * overlap_ablations + [False] * non_overlap_ablations + [True] * non_existing_ablations,
        )

        task["true_labels"] = true_labels
        task["pred_labels"] = pred_labels
        task["precision"] = precision_score(true_labels, pred_labels)
        task["recall"] = recall_score(true_labels, pred_labels)
        task["f1_score"] = f1_score(true_labels, pred_labels)
        task["cost"] = cost
        return task

    def evaluate(self, predictions_path: Path, dataset: Dataset, top_k: int | None) -> EvaluationResult:
        """Evaluate predictions against ground truth."""
        dataset_full_name = f"ai-coscientist/{dataset.info.dataset_name}"
        with_labels = dataset.map(
            lambda task: self.map_func[dataset_full_name](
                task, predictions_path, top_k, self.map_type[dataset_full_name]
            ),
            remove_columns=[column for column in dataset.column_names if column != "id"],
            desc="Evaluating instances using LM judge",
            num_proc=self.config.parallelism,
        )
        with_labels_df = with_labels.to_pandas()
        with_labels_df.to_json(self.config.output_dir / "evaluations.json", orient="records", indent=4)

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
