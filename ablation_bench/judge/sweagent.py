import io
import json
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score

from ablation_bench.judge.abstract import Judge, register_judge
from ablation_bench.logger import get_logger
from ablation_bench.types import (
    AblationSuggestionPred,
    DatasetForEvaluation,
    EvaluationResult,
    MissingAblationSuggestionPred,
    SingleResult,
)


@register_judge("sweagent")
class SweAgentJudge(Judge):
    """Judge class for evaluating SWE agent ablations."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the SweAgentJudge.

        Args:
            config: A dictionary containing the judge-specific configuration.
                    Expected keys include 'output_dir', 'agent' (sub-dict),
                    and 'num_workers'.
        """
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.map_func = {
            DatasetForEvaluation.ResearcherAssist.value: self._process_predictions_researcherassist,
            DatasetForEvaluation.ReviewerAssist.value: self._process_predictions_reviewerassist,
        }

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        model_name: str,
        output_dir: Path,
        parallelism: int = 1,
        reasoning_effort: str | None = None,
    ) -> "Judge":
        """Create a judge instance from a config file.

        Args:
            config_path: Path to the configuration file.
            model_name: Name of the model to be used by the agent.
            output_dir: Directory to save outputs.
            parallelism: Number of parallel workers to use.

        Returns:
            A configured SweAgentJudge instance.
        """
        config = yaml.safe_load(config_path.read_text())
        config["output_dir"] = output_dir.absolute()
        completion_kwargs = {"drop_params": True}
        if reasoning_effort:
            completion_kwargs["reasoning_effort"] = reasoning_effort
        config["agent"]["model"] = {"name": model_name, "completion_kwargs": completion_kwargs, "delay": 0.0}
        config["num_workers"] = parallelism
        return cls(config=config)

    def _convert_to_sweagent_instances_researcherassist(
        self, dataset: Dataset, plans_path: Path
    ) -> list[dict[str, Any]]:
        """Convert dataset tasks to SWE agent instances format."""
        for_swe_agent = []

        for task in dataset:
            name = task["id"]
            d = defaultdict(lambda: defaultdict(dict))

            # Set environment configuration
            d["env"]["deployment"]["type"] = "docker"
            d["env"]["deployment"]["image"] = "ablations-bench:judge"
            d["env"]["deployment"]["docker_args"] = ["-u", "root"]
            d["env"]["repo"]["type"] = "preexisting"
            d["env"]["repo"]["repo_name"] = "repo"

            # Set problem statement
            plan = (plans_path / f"{name}.jsonl").read_text()
            plan = [p for p in plan.split("\n") if p.strip() != ""]
            random.shuffle(plan)
            plan = "\n".join(plan)
            ablations_in_paper = json.loads(task["ablations_in_paper"])
            random.shuffle(ablations_in_paper)
            ablations_in_paper = "\n".join([json.dumps(ablation) for ablation in ablations_in_paper])
            sides = [ablations_in_paper, plan]
            random.shuffle(sides)

            d["problem_statement"]["type"] = "text"
            d["problem_statement"]["text"] = task["ablations_in_paper"]
            d["problem_statement"]["id"] = name
            d["problem_statement"]["extra_fields"] = {
                "paper_title": task["paper_title"],
                "abstract": task["paper_abstract"],
                "side_A": sides[0],
                "side_B": sides[1],
            }
            for_swe_agent.append(json.loads(json.dumps(d)))

        return for_swe_agent

    def _convert_to_sweagent_instances_reviewerassist(self, dataset: Dataset, plans_path: Path) -> list[dict[str, Any]]:
        """Convert dataset tasks to SWE agent instances format."""
        for_swe_agent = []

        for task in dataset:
            name = task["id"]
            d = defaultdict(lambda: defaultdict(dict))

            # Set environment configuration
            d["env"]["deployment"]["type"] = "docker"
            d["env"]["deployment"]["image"] = "ablations-bench:judge"
            paper_path = (Path("data/papers/full") / name).absolute()
            assert paper_path.is_dir(), f"Paper path {paper_path} does not exist."
            d["env"]["deployment"]["docker_args"] = ["-u", "root", "-v", f"{paper_path}:/paper:ro"]
            d["env"]["repo"]["type"] = "preexisting"
            d["env"]["repo"]["repo_name"] = "repo"

            # Set problem statement
            d["problem_statement"]["type"] = "text"
            plan = (plans_path / f"{name}.jsonl").read_text()
            plan = [p for p in plan.split("\n") if p.strip() != ""]
            random.shuffle(plan)
            plan = "\n".join(plan)
            d["problem_statement"]["text"] = plan
            d["problem_statement"]["id"] = name
            reviews = json.loads(task["review_text"])
            random.shuffle(reviews)
            d["problem_statement"]["extra_fields"] = {
                "paper_title": task["paper_title"],
                "official_reviews": "\n</official_review>\n\n\n<official_review>\n".join(reviews),
                "abstract": task["paper_abstract"],
            }

            for_swe_agent.append(json.loads(json.dumps(d)))

        return for_swe_agent

    def _convert_to_sweagent_instances(self, dataset: Dataset, plans_path: Path) -> list[dict[str, Any]]:
        """Convert a dataset into a list of instances formatted for SWE-agent.

        This method dispatches to specific conversion functions based on the dataset type
        (ResearcherAssist or ReviewerAssist).

        Args:
            dataset: The input dataset to convert.
            plans_path: Path to the directory containing ablation plans.

        Returns:
            A list of dictionaries, each representing an instance for SWE-agent.

        Raises:
            ValueError: If the dataset type is unknown.
        """
        dataset_full_name = f"ai-coscientist/{dataset.info.dataset_name}"
        if dataset_full_name == DatasetForEvaluation.ResearcherAssist.value:
            return self._convert_to_sweagent_instances_researcherassist(dataset, plans_path)
        elif dataset_full_name == DatasetForEvaluation.ReviewerAssist.value:
            return self._convert_to_sweagent_instances_reviewerassist(dataset, plans_path)
        raise ValueError(f"Unknown dataset: {dataset_full_name}")

    def _process_predictions_researcherassist(
        self, task: dict[str, Any], preds: dict, predictions_path: Path, top_k: int | None
    ) -> dict[str, Any]:
        """Process SWE-agent predictions for a single task in ResearcherAssist mode.

        This involves parsing the agent\'s output (often a patch), comparing it to
        ground truth ablations from the paper and the original plan, and calculating
        evaluation metrics (precision, recall, F1).

        Args:
            task: The task dictionary containing ground truth information.
            preds: A dictionary of predictions from the SWE-agent, keyed by task ID.
            predictions_path: Path to the directory of original ablation plans.
            top_k: If specified, consider only the top K ablations from the plan.

        Returns:
            The task dictionary updated with evaluation results.
        """
        task_id = task["id"]
        ablations_in_paper = pd.read_json(io.StringIO(task["ablations_in_paper"]))
        ablations_in_plan = pd.read_json(predictions_path / f"{task_id}.jsonl", lines=True)
        if top_k:
            ablations_in_plan = ablations_in_plan.head(top_k)
        if task_id not in preds:
            # If no predictions for this task, mark all paper ablations as false negatives
            task["true_labels"] = [True] * len(ablations_in_paper) + [False] * len(ablations_in_plan)
            task["pred_labels"] = [False] * len(ablations_in_paper) + [True] * len(ablations_in_plan)
            return task

        # Parse predictions for this task
        model_patch = preds[task_id]["model_patch"]
        if model_patch.startswith("diff --git"):
            try:
                model_patch = subprocess.check_output(
                    ["patch", "-p1", "-i-", "-o-"], input=model_patch.encode(), stderr=subprocess.DEVNULL
                ).decode()
            except Exception as ex:
                self.logger.error(f"Error applying patch for task {task_id}: {ex}")
                model_patch = (
                    "\n".join(
                        [
                            AblationSuggestionPred(name_in_paper=ablation["name"]).model_dump_json()
                            for _, ablation in ablations_in_paper.iterrows()
                        ]
                    )
                    .replace("name_in_paper", "name_in_A")
                    .replace("name_in_plan", "name_in_B")
                )

        try:
            model_patch = pd.read_json(io.StringIO(model_patch), lines=True).replace({float("nan"): None})
            model_patch = model_patch.loc[model_patch.astype(str).drop_duplicates().index]
            name_in_a = set(model_patch["name_in_A"].explode().dropna().to_list())
            name_in_b = set(model_patch["name_in_B"].explode().dropna().to_list())
            if name_in_a <= (set(ablations_in_paper["name"].to_list())):
                model_patch = model_patch.rename(columns={"name_in_A": "name_in_paper", "name_in_B": "name_in_plan"})
            elif name_in_b <= (set(ablations_in_paper["name"].to_list())):
                model_patch = model_patch.rename(columns={"name_in_B": "name_in_paper", "name_in_A": "name_in_plan"})
            else:
                raise ValueError("Could not determine name mapping between paper and plan ablations.")
        except Exception as ex:
            print(f"Error processing model patch JSON: {ex}, creating an empty model patch")
            model_patch = pd.read_json(
                io.StringIO(
                    "\n".join(
                        [
                            AblationSuggestionPred(name_in_paper=ablation["name"]).model_dump_json()
                            for _, ablation in ablations_in_paper.iterrows()
                        ]
                    )
                ),
                lines=True,
            ).replace({float("nan"): None})

        model_patch = model_patch[model_patch["name_in_paper"].notna()]
        model_patch = model_patch.explode("name_in_paper")
        model_patch = model_patch[model_patch["name_in_paper"].notna()]
        model_patch.to_json(self.config["output_dir"] / f"{task_id}.jsonl", orient="records", lines=True)
        ablation_preds = [AblationSuggestionPred(**pred) for pred in model_patch.to_dict(orient="records")]

        true_labels, pred_labels = self._get_labels(
            predictions=ablation_preds,
            ablations_in_paper=[
                str(ablation.get("name"))
                for ablation in ablations_in_paper.replace({float("nan"): None}).to_dict(orient="records")
            ],
            ablations_in_plan=[
                str(ablation.get("name"))
                for ablation in ablations_in_plan.replace({float("nan"): None}).to_dict(orient="records")
            ],
        )

        cost = 0.0
        traj_path = self.config["output_dir"] / task_id / f"{task_id}.traj"
        if traj_path.is_file():
            traj_data = json.loads(traj_path.read_text())
            cost = traj_data["info"]["model_stats"]["instance_cost"]

        task["true_labels"] = true_labels
        task["pred_labels"] = pred_labels
        task["precision"] = precision_score(true_labels, pred_labels)
        task["recall"] = recall_score(true_labels, pred_labels)
        task["f1_score"] = f1_score(true_labels, pred_labels)
        task["cost"] = cost
        return task

    def _process_predictions_reviewerassist(
        self, task: dict[str, Any], preds: dict, predictions_path: Path, top_k: int | None
    ) -> dict[str, Any]:
        """Process SWE-agent predictions for a single task in ReviewerAssist mode.

        This involves parsing the agent\'s output to determine which planned ablations
        were identified as appearing in reviews, then calculating metrics.

        Args:
            task: The task dictionary containing ground truth (e.g., number of suggestions).
            preds: A dictionary of predictions from the SWE-agent, keyed by task ID.
            predictions_path: Path to the directory of original ablation plans.
            top_k: If specified, consider only the top K ablations from the plan.

        Returns:
            The task dictionary updated with evaluation results.
        """
        task_id = task["id"]

        # Parse predictions for this task
        model_patch = preds[task_id]["model_patch"]
        if model_patch.startswith("diff --git"):
            try:
                model_patch = subprocess.check_output(
                    ["patch", "-p1", "-i-", "-o-"], input=model_patch.encode(), stderr=subprocess.DEVNULL
                ).decode()
            except Exception as ex:
                self.logger.error(f"Error applying patch for task {task_id}: {ex}")
                ablations_in_plan = json.loads((predictions_path / f"{task_id}.jsonl").read_text())
                model_patch = "\n".join(
                    [
                        MissingAblationSuggestionPred(name_in_plan=ablation["name"]).model_dump_json()
                        for ablation in ablations_in_plan
                    ]
                )

        model_patch = pd.read_json(io.StringIO(model_patch), lines=True).replace({float("nan"): None}).drop_duplicates()
        model_patch.to_json(self.config["output_dir"] / f"{task_id}.jsonl", orient="records", lines=True)
        ablation_preds = [MissingAblationSuggestionPred(**pred) for pred in model_patch.to_dict(orient="records")]

        if top_k:
            ablation_preds = ablation_preds[:top_k]

        num_ablations = task["num_ablation_suggestions"]
        overlap_ablations = min(sum([pred.appears_in_review for pred in ablation_preds]), num_ablations)
        non_overlap_ablations = max(0, num_ablations - overlap_ablations)
        non_existing_ablations = sum([not pred.appears_in_review for pred in ablation_preds])

        true_labels, pred_labels = (
            [True] * num_ablations + [False] * non_existing_ablations,
            [True] * overlap_ablations + [False] * non_overlap_ablations + [True] * non_existing_ablations,
        )

        cost = 0.0
        traj_path = self.config["output_dir"] / task_id / f"{task_id}.traj"
        if traj_path.is_file():
            traj_data = json.loads(traj_path.read_text())
            cost = traj_data["info"]["model_stats"]["instance_cost"]

        task["true_labels"] = true_labels
        task["pred_labels"] = pred_labels
        task["precision"] = precision_score(true_labels, pred_labels)
        task["recall"] = recall_score(true_labels, pred_labels)
        task["f1_score"] = f1_score(true_labels, pred_labels)
        task["cost"] = cost
        return task

    def _process_predictions(
        self, preds: dict, predictions_path: Path, dataset: Dataset, top_k: int | None
    ) -> pd.DataFrame:
        """Process predictions and create true/predicted label lists."""

        with_labels = dataset.map(
            lambda task: self.map_func[dataset.info.dataset_name](task, preds, predictions_path, top_k),
            remove_columns=[column for column in dataset.column_names if column != "id"],
            desc="Calculating metrics using SWE agent judge",
            num_proc=self.config["num_workers"],
        )
        return with_labels.to_pandas()

    def evaluate(self, predictions_path: Path, dataset: Dataset, top_k: int | None) -> EvaluationResult:
        """Evaluate predictions against ground truth."""

        # Convert dataset to SWE agent instances
        instances = self._convert_to_sweagent_instances(dataset, predictions_path)

        # Save instances to YAML file
        instances_path = self.config["output_dir"] / "sweagent_instances.yaml"
        with instances_path.open("w") as f:
            yaml.safe_dump(instances, f)

        # Run SWE agent evaluation
        self.config.update({"instances": {"type": "expert_file", "path": str(instances_path.absolute())}})
        from sweagent import REPO_ROOT
        from sweagent.run.run_batch import RunBatchConfig, run_from_config

        run_config = RunBatchConfig(**self.config)
        cur_dir = Path.cwd()
        os.chdir(REPO_ROOT)  # We need this because sweagent can't run from within another package
        run_from_config(run_config)
        os.chdir(cur_dir)

        # Calculate evaluation metrics
        preds = json.loads((run_config.output_dir / "preds.json").read_text())
        all_metrics = self._process_predictions(preds, predictions_path, dataset, top_k)

        all_metrics.to_json(run_config.output_dir / "evaluations.json", orient="records", indent=4)

        result = all_metrics[["precision", "recall", "f1_score", "cost"]].mean()
        result_std_dev = all_metrics[["precision", "recall", "f1_score"]].std()

        return EvaluationResult(
            precision=SingleResult(result=result.precision, std_dev=result_std_dev.precision),
            recall=SingleResult(result=result.recall, std_dev=result_std_dev.recall),
            f1_score=SingleResult(result=result.f1_score, std_dev=result_std_dev.f1_score),
            cost=result.cost,
        )
