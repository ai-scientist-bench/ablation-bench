"""Main CLI entrypoint for ablations-bench."""

import typer
from dotenv import load_dotenv
from rich import print

from ablation_bench.harness.evaluation import app as eval_app
from ablation_bench.harness.judge_evaluation import app as judge_eval_app
from ablation_bench.planner.plan import app as plan_app

app = typer.Typer(
    name="ablation-bench", help="Benchmarking tool for ablation planning in empirical AI research", no_args_is_help=True
)

# Add subcommands
app.add_typer(eval_app, name="eval", help="Run ablation evaluation")
app.add_typer(judge_eval_app, name="eval-judge", help="Run judge evaluation")
app.add_typer(plan_app, name="plan", help="Run ablation planning")


def main() -> None:
    """Main entrypoint."""
    print(
        "[bold medium_orchid1 on green]"
        "🩺 Welcome to AblationBench: A benchmark suite for ablation planning using LM agents!"
        "[/bold medium_orchid1 on green]"
    )
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
