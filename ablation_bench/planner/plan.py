"""Planning module for ablations-bench."""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from datasets import Dataset, load_dataset
from pydantic import Field
from pydantic_settings import BaseSettings

from ablation_bench.harness.utils import get_field_description
from ablation_bench.logger import get_logger
from ablation_bench.planner import PlannerType
from ablation_bench.planner.abstract import Planner
from ablation_bench.types import DatasetForEvaluation, DatasetSplit

app = typer.Typer(name="plan", help="Run ablation planning", no_args_is_help=True)


class PlannerSettings(BaseSettings):
    """Settings for planning run."""

    planner: PlannerType = Field(description="Name of the planner class to use")  # type: ignore
    planner_config: Path = Field(description="Path to the planner configuration file (YAML format)")
    model_name: str = Field(
        description="Name of the model used to plan (e.g., 'openai/gpt-4'). Any LiteLLM supported model is valid."
    )
    dataset: DatasetForEvaluation = Field(
        description="Name of the dataset in HuggingFace hub (e.g., 'tau/ablations-bench')"
    )
    split: DatasetSplit = Field(default="dev", description="Dataset split to use for planning (e.g., 'dev', 'test')")
    output_dir: Path | None = Field(default=None, description="Path to the directory to save generated plans")
    parallelism: int = Field(default=1, description="Number of parallel workers to use for planning")
    num_ablations: int = Field(default=5, description="Number of ablations to generate per paper")

    class Config:
        env_prefix = "ABLATIONS_"


class Runner:
    """Main runner class for generating ablation plans."""

    def __init__(self, settings: PlannerSettings):
        """Initialize the Runner with planning settings.

        Args:
            settings: The PlannerSettings object containing all configurations.
        """
        self.settings = settings
        self.dataset: Dataset | None = None
        self.planner: Planner | None = None
        self.logger = get_logger(__name__)

        # Create output directory
        if settings.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            settings.output_dir = (Path("runs") / timestamp).absolute()

        settings.output_dir.mkdir(parents=True, exist_ok=True)

    def load_planner(self) -> Planner:
        """Load planner class and instantiate with config."""
        self.logger.info(
            f"Loading planner {self.settings.planner.value} "
            f"and instantiating with config {self.settings.planner_config}"
        )
        planner_class = Planner.get_planner(self.settings.planner.value)
        self.planner = planner_class.from_config(
            self.settings.planner_config,
            model_name=self.settings.model_name,
            parallelism=self.settings.parallelism,
            num_ablations=self.settings.num_ablations,
            output_dir=self.settings.output_dir,
        )
        return self.planner

    def load_data(self) -> Dataset:
        """Load dataset from Hugging Face."""
        self.logger.info(f"Loading dataset {self.settings.dataset.value}, split {self.settings.split.value}")
        self.dataset = load_dataset(self.settings.dataset.value, split=self.settings.split.value)
        return self.dataset

    def run_planning(self) -> None:
        """Run the complete planning process."""
        # Load dataset
        self.load_data()

        # Initialize planner
        self.load_planner()

        # Run planning
        self.planner.plan(dataset=self.dataset)

        self.logger.info(f"Planning completed. Results saved to {self.settings.output_dir}")


@app.callback(invoke_without_command=True)
def plan(
    planner: Annotated[PlannerType, typer.Option(help=get_field_description(PlannerSettings, "planner"))],
    planner_config: Annotated[Path, typer.Option(help=get_field_description(PlannerSettings, "planner_config"))],
    model_name: Annotated[str, typer.Option(help=get_field_description(PlannerSettings, "model_name"))],
    dataset: Annotated[DatasetForEvaluation, typer.Option(help=get_field_description(PlannerSettings, "dataset"))],
    split: Annotated[DatasetSplit, typer.Option(help=get_field_description(PlannerSettings, "split"))],
    parallelism: Annotated[int, typer.Option(help=get_field_description(PlannerSettings, "parallelism"))] = 1,
    num_ablations: Annotated[int, typer.Option(help=get_field_description(PlannerSettings, "num_ablations"))] = 5,
    output_dir: Annotated[Path | None, typer.Option(help=get_field_description(PlannerSettings, "output_dir"))] = None,
) -> None:
    """Run ablation planning with the specified settings."""
    settings = PlannerSettings(
        planner=planner,
        planner_config=planner_config,
        dataset=dataset,
        split=split,
        model_name=model_name,
        parallelism=parallelism,
        num_ablations=num_ablations,
        output_dir=output_dir,
    )
    runner = Runner(settings)
    runner.run_planning()
