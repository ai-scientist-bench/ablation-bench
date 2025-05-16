import json
import logging
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset
from jinja2 import Template
from litellm import ModelResponse, completion, completion_cost
from tenacity import retry, stop_after_attempt, wait_exponential

from ablation_bench.logger import get_logger
from ablation_bench.planner.abstract import Planner, register_planner
from ablation_bench.types import AblationPlanPredResponse, AblationPlanSimpleLMConfig

TEXT_FILE_EXT = ["*.tex", "*.text", ".txt", "*.bib", "*.bbl", "*.md"]


@register_planner("simple_lm")
class SimpleLMPlanner(Planner):
    """Planner that uses a language model to generate ablation plans."""

    def __init__(self, config: AblationPlanSimpleLMConfig) -> None:
        """Initialize the SimpleLMPlanner.

        Args:
            config: The configuration object for the planner.
        """
        super().__init__(config)
        self.system_prompt = config.prompts["system"]
        self.user_prompt = config.prompts["user"]
        self.logger = get_logger(__name__)

    @classmethod
    def from_config(
        cls, config_path: Path, model_name: str, output_dir: Path, num_ablations: int = 5, parallelism: int = 1
    ) -> "Planner":
        """Create planner from config file."""
        config = yaml.safe_load(config_path.read_text())
        config["model"] = {"name": model_name}
        config["output_dir"] = output_dir
        config["parallelism"] = parallelism
        config["num_ablations"] = num_ablations
        config = AblationPlanSimpleLMConfig(**config)
        return cls(config=config)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=lambda retry_state: logging.warning(
            f"Retry attempt {retry_state.attempt_number} after error: {retry_state.outcome.exception()}"
        ),
    )
    def _get_lm_response(self, task_id: str, **kwargs: dict[str, Any]) -> AblationPlanPredResponse:
        """Get LM response for a single task."""
        log_path = self.config.output_dir / f"{task_id}.json.log"
        if log_path.is_file():
            try:
                response_dict = json.loads(log_path.read_text())
                response = ModelResponse(**response_dict)
                pred_response = AblationPlanPredResponse.from_lm_response(
                    response.choices[0].message.content, completion_cost(response)
                )
                return pred_response
            except Exception as ex:
                self.logger.warning(f"Failed to read log file {log_path}: {ex}")

        user_prompt_template = Template(self.user_prompt)
        formatted_user_prompt = user_prompt_template.render(**kwargs)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": formatted_user_prompt},
        ]
        debug_log = log_path.with_suffix(".debug.log")
        debug_log.write_text(json.dumps(messages, indent=4))

        response = completion(
            model=self.config.model.name,
            messages=messages,
            temperature=self.config.model.temperature,
            top_p=self.config.model.top_p,
        )
        log_path.write_text(json.dumps(response.to_dict(mode="json"), indent=4))
        pred_response = AblationPlanPredResponse.from_lm_response(
            response.choices[0].message.content, completion_cost(response)
        )
        return pred_response

    def _get_paper_source(self, paper_path: Path) -> str:
        """Get the source of the paper without ablations."""

        paper_source = ""
        for extension in TEXT_FILE_EXT:
            for file_path in paper_path.rglob(extension):
                paper_source += f'<file name="{file_path.relative_to(paper_path)}">\n'
                paper_source += file_path.read_text(encoding="utf-8")
                paper_source += "\n</file>\n"

        return paper_source

    def _process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process a research assist task."""
        task["predictions"] = ""
        task["cost"] = 0.0
        try:
            response = self._get_lm_response(
                task_id=task["id"],
                paper_title=task["paper_title"],
                problem_statement=task["paper_abstract"],
                num_ablations=self.config.num_ablations,
                paper_source=self._get_paper_source(Path(task["paper_path"])),
            )
            predictions = "\n".join(pred.model_dump_json() for pred in response.predictions)
            output_path = self.config.output_dir / f"{task['id']}.jsonl"
            output_path.write_text(predictions)
            task["predictions"] = predictions
            task["cost"] = response.cost
        except Exception as ex:
            self.logger.warning(f"Failed to process task {task['id']}: {ex}")
            output_path = self.config.output_dir / f"{task['id']}.jsonl"
            output_path.write_text("")
        return task

    def plan(self, dataset: Dataset) -> None:
        """Plan ablations using language model."""

        dataset_with_preds = dataset.map(
            self._process_task,
            desc="Planning ablations using LM",
            num_proc=self.config.parallelism,
        )

        output_path = self.config.output_dir / "plans.json"
        dataset_with_preds.to_pandas()[["id", "predictions", "cost"]].set_index("id").to_json(
            output_path, orient="index", indent=4
        )
