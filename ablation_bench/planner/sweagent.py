import json
import os
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import Dataset

from ablation_bench.logger import get_logger
from ablation_bench.planner.abstract import Planner, register_planner


@register_planner("sweagent")
class SWEAgentPlanner(Planner):
    """Planner that uses SWE agent to generate ablation plans."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the SWEAgentPlanner.

        Args:
            config: A dictionary containing the planner-specific configuration.
                    Expected keys include 'output_dir', 'agent' (sub-dict),
                    'num_workers', and optionally 'num_ablations'.
        """
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.num_ablations = config.pop("num_ablations", 5)

    @classmethod
    def from_config(
        cls, config_path: Path, model_name: str, output_dir: Path, num_ablations: int = 5, parallelism: int = 1
    ) -> "Planner":
        """Create planner from config file."""
        config = yaml.safe_load(config_path.read_text())
        config["output_dir"] = output_dir.absolute()
        config["agent"]["model"] = {"name": model_name}
        config["num_workers"] = parallelism
        config["num_ablations"] = num_ablations
        return cls(config=config)

    def _convert_to_sweagent_instances(self, dataset: Dataset) -> list[dict[str, Any]]:
        """Convert dataset tasks to SWE agent instances format."""
        instances = []
        for task in dataset:
            d = {
                "env": {
                    "deployment": {"type": "docker", "image": task["docker_image"], "docker_args": ["-u", "root"]},
                    "repo": {
                        "type": "preexisting",
                        "repo_name": "repo",
                    },
                },
                "problem_statement": {
                    "type": "text",
                    "text": task["paper_abstract"],
                    "id": task["id"],
                    "extra_fields": {
                        "paper_title": task["paper_title"],
                        "num_ablations": self.num_ablations,
                    },
                },
            }
            instances.append(d)
        return instances

    def _calc_cost(self, task_id: str) -> float:
        """Calculate the cost associated with a task based on its trajectory file.

        Args:
            task_id: The ID of the task.

        Returns:
            The cost extracted from the trajectory file, or 0.0 if not found or error.
        """
        cost = 0.0
        traj_path = self.config["output_dir"] / task_id / f"{task_id}.traj"
        if traj_path.is_file():
            try:
                traj_data = json.loads(traj_path.read_text())
                cost = traj_data.get("info", {}).get("model_stats", {}).get("instance_cost", 0.0)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding JSON from trajectory file {traj_path}: {e}")
            except Exception as e:
                self.logger.error(f"Error reading cost from trajectory file {traj_path}: {e}")
        return cost

    def _write_to_jsonl(self, row: dict[str, Any]) -> None:
        """Process predictions for a row and write to a .jsonl file.

        If predictions start with 'diff --git', they are treated as a patch
        and applied. Duplicate lines are removed while preserving order.

        Args:
            row: A dictionary representing a row of predictions, expected to have
                 'predictions' and 'instance_id' keys.
        """
        predictions = row.get("predictions", "")
        instance_id = row.get("instance_id")

        if not instance_id:
            self.logger.error("'instance_id' not found in row, cannot write .jsonl file.")
            return

        output_path = self.config["output_dir"] / f"{instance_id}.jsonl"

        try:
            if predictions.startswith("diff --git"):
                # Ensure predictions is a string before encoding
                predictions_str = predictions if isinstance(predictions, str) else ""
                patched_predictions = subprocess.check_output(
                    ["patch", "-p1", "-i-", "-o-"], input=predictions_str.encode(), stderr=subprocess.PIPE
                ).decode()
                # Remove duplicate lines while preserving order
                predictions = "\n".join(list(OrderedDict.fromkeys(patched_predictions.splitlines())))

            output_path.write_text(predictions)

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error applying patch for instance {instance_id}: {e}\nStderr: {e.stderr.decode()}")
            # Optionally write empty or original predictions on error
            output_path.write_text("")  # Or write `row.get("predictions", "")`
        except Exception as e:
            self.logger.error(f"Error in _write_to_jsonl for instance {instance_id}: {e}")
            output_path.write_text("")  # Fallback

    def plan(self, dataset: Dataset) -> None:
        """Plan ablations using SWE agent."""
        instances = self._convert_to_sweagent_instances(dataset)
        instances_path = self.config["output_dir"] / "sweagent_instances.yaml"
        instances_path.write_text(yaml.safe_dump(instances))

        self.config.update({"instances": {"type": "expert_file", "path": str(instances_path.absolute())}})

        from sweagent import REPO_ROOT
        from sweagent.run.run_batch import RunBatchConfig, run_from_config

        run_config = RunBatchConfig(**self.config)

        cur_dir = Path.cwd()
        os.chdir(REPO_ROOT)  # Required because sweagent can't run from within another package
        run_from_config(run_config)
        os.chdir(cur_dir)

        # Save the output to the specified output directory
        preds = pd.read_json(self.config["output_dir"] / "preds.json", orient="index")
        preds.rename(columns={"model_patch": "predictions"}, inplace=True)
        preds.apply(lambda row: self._write_to_jsonl(row), axis=1)
        preds["cost"] = preds.apply(lambda row: self._calc_cost(row["instance_id"]), axis=1)
        preds.drop(columns=["model_name_or_path", "instance_id"], inplace=True)
        preds.to_json(self.config["output_dir"] / "plans.json", orient="index", indent=4)
