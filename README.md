# AblationBench: Evaluating Language Models on Ablation Planning in Empirical AI Research

AblationBench is a comprehensive benchmarking tool designed to evaluate and facilitate the creation of ablation plans, particularly in the context of AI and Machine Learning research. It provides a standardized framework for generating ablation plans and evaluating their quality using different LM-based judges, including chain-of-thought (CoT) prompting and using [SWE-agent](https://github.com/SWE-agent/SWE-agent) based judge.


## 🛠️ Installation

1.  **Prerequisites**:
    *   Python 3.11 or higher.
    *   `git-lfs` for cloning the repository data files.
    *   Docker (if using SWE-agent based planners or judges).

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/ai-scientist-bench/ablation-bench.git
    cd ablation-bench
    git lfs fetch --all
    git lfs pull
    ```

3.  **Create and activate a virtual environment (recommended)**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    # On Windows: .venv\Scripts\activate
    ```

4.  **Install the package and its dependencies**:
    ```bash
    pip install -e .
    ```
    This installs the package in editable mode. If you prefer a standard installation, use `pip install .`.

5.  **(Optional) Install SWE-agent**:

    If you wish to use the agent judge or planner, please install SWE-agent from source.
    You should clone the SWE-agent repository and run the installation:
    ```bash
    git clone https://github.com/SWE-agent/SWE-agent.git
    cd SWE-agent/
    git checkout v1.0.1
    pip install -e .
    ```
    Then you should copy the tools provided in `config/sweagent/tools` to SWE-agent directory:
    ```bash
    cp -R ../ablation-bench/config/sweagent/tools/* tools/
    ```
    For further details about SWE-agent installation please refer to its [documentation](https://swe-agent.com/latest/).

## 🤗 Datasets

The benchmark utilizes datasets hosted on Hugging Face:

*   `ai-coscientist/researcher-ablation-bench`: For ablation planning and evaluation given paper's method section (author-assist mode).
*   `ai-coscientist/reviewer-ablation-bench`: For tasks related to identifying missing ablations given a full paper (reviewer-assist mode).
*   `ai-coscientist/researcher-ablation-judge-eval`: For evaluating the performance of different judges for the ReviewerAblationBench.
*   `ai-coscientist/reviewer-ablation-judge-eval`: For evaluating the performance of different judges for the ResearcherAblationBench.


## ⚙️ Configuration


### 🔑 API Keys

Store your API keys (e.g., OpenAI API key or OpenRouter API key) in a `.env` file in the project root directory. AblationBench uses `python-dotenv` to load these variables. Example `.env` file:
```env
OPENAI_API_KEY="your_openai_api_key_here"
OPENROUTER_API_KEY="your_openrouter_api_key_here"
HF_TOKEN="your_hf_token_here"
```

### Planners and Judges

Planners and judges in AblationBench are configured using YAML files. You will need to either use the provided configuration in `config/` directory or create these configuration files based on your needs.

*   **Planner Configuration**: Defines parameters for the chosen planner (e.g., prompts for `SimpleLMPlanner`, agent settings for `SWEAgentPlanner`).
    *   The LM-Planner configuration contains mainly the system and user prompts used for each task.
    *   The SWE-agent configuration contains many different parameters used to control the agent. You can read more about them [here](https://swe-agent.com/latest/config/).
*   **Judge Configuration**: Defines parameters for the chosen judge (e.g., prompts for `SimpleLMJudge`, agent settings for `SweAgentJudge`).
    *   The LMJudge configuration contains mainly the system and user prompts used for each judge.
    *   The SWE-agent configuration contains many different parameters used to control the agent. You can read more about them [here](https://swe-agent.com/latest/config/).


## 🏃🏾 Quick Start


### 👩‍🔬 ResearcherAblationBench

Here's a quick start example of how to generate an ablation plan using the LM-Planner, then evaluate it using the LMJudge:


1.  **Run the planning command**:
    ```bash
    ablation-bench plan \
        --planner simple_lm \
        --planner-config config/simple_lm/plan_ablations.yaml \
        --model-name "openrouter/openai/o3-mini-high" \
        --dataset "ai-coscientist/researcher-ablation-bench" \
        --split "test" \
        --num-ablations 5 \
        --output-dir "./runs/plan/researcher/simple_lm/o3-mini-high"
    ```
    This will generate ablation plans for all the papers in the test set of ResearcherAblationBench. Each plan will contain up to 5 generated ablation studies using o3-mini-high. The command will save these to the `./runs/plan/researcher/simple_lm/o3-mini-high` directory.

2.  **Run the evaluation command**:
    ```bash
    ablation-bench eval \
        --judge simple_lm \
        --judge-config config/simple_lm/judge_ablation_suggestions.yaml \
        --model-name "openrouter/openai/gpt-4o" \
        --dataset "ai-coscientist/researcher-ablation-bench" \
        --split "test" \
        --generated-plans-path "./runs/researcher/plan/simple_lm/o3-mini-high" \
        --parallelism 5 \
        --output-dir "./runs/eval/researcher/simple_lm/o3-mini-high"
    ```
    This will evaluate ablation plans generated in the previous step, for all the papers in the test set of ResearcherAblationBench. The command will save evaluation results to the `./runs/eval/researcher/simple_lm/o3-mini-high` directory, and print them to the screen as well.

### 🔍 ReviewerAblationBench

Here's a quick start example of how to generate a missing ablation plan using the LM-Planner, then evaluate it using the LMJudge:

1.  **Run the planning command**:
    ```bash
    ablation-bench plan \
        --planner simple_lm \
        --planner-config config/simple_lm/plan_missing_ablations.yaml \
        --model-name "openrouter/openai/o3-mini-high" \
        --dataset "ai-coscientist/reviewer-ablation-bench" \
        --split "test" \
        --num-ablations 2 \
        --output-dir "./runs/plan/reviewer/simple_lm/o3-mini-high"
    ```
    This will generate ablation plans for all the papers in the test set of ReviewerAblationBench. Each plan will contain up to 2 generated ablation studies using o3-mini-high. The command will save these to the `./runs/plan/reviewer/simple_lm/o3-mini-high` directory.

2.  **Run the evaluation command**:
    ```bash
    ablation-bench eval \
        --judge simple_lm \
        --judge-config config/simple_lm/judge_missing_ablation_suggestions.yaml \
        --model-name "openrouter/anthropic/claude-3.5-sonnet" \
        --dataset "ai-coscientist/reviewer-ablation-bench" \
        --split "test" \
        --generated-plans-path "./runs/plan/reviewer/simple_lm/o3-mini-high" \
        --parallelism 5 \
        --output-dir "./runs/eval/reviewer/simple_lm/o3-mini-high"
    ```
    This will evaluate ablation plans generated in the previous step, for all the papers in the test set of ResearcherAblationBench. The command will save evaluation results to the `./runs/eval/reviewer/simple_lm/o3-mini-high` directory, and print them to the screen as well.


## 🫀 Core Components

AblationBench comprises two main types of components: **Planners** and **Judges**.

### ✍️ Planners

Planners are responsible for generating ablation study plans based on the benchmark input.

*   **`SimpleLMPlanner`**:
    *   Uses a Large Language Model (via LiteLLM) to generate ablation plans.
    *   Highly configurable through prompts defined in its YAML configuration file.
    *   Requires an LM API key (e.g., OpenAI).
    *   Configuration files for the LM-Planner can be found in `config/simple_lm/plan_*.yaml`
*   **`SWEAgentPlanner`**:
    *   Utilizes the SWE-agent framework to devise ablation plans.
    *   Interacts with code repositories and executes tasks in a Docker environment.
    *   Configuration involves SWE-agent specific parameters in its YAML file and requires Docker.
    *   Configuration files for the Agent-Planner can be found in `config/sweagent/plan_*.yaml`

### 🧑‍⚖️ Judges

Judges are used to evaluate the quality of ablation plans or to perform other assessment tasks, such as identifying if a planned ablation was mentioned in a peer review.

*   **`SimpleLMJudge`**:
    *   Employs an LM (via LiteLLM) to evaluate ablation suggestions or plans against certain criteria (e.g., relevance to a paper, presence in reviews).
    *   Its behavior is controlled by prompts in its YAML configuration file.
    *   Requires an LM API key.
    *   Configuration files for the LMJudge can be found in `config/simple_lm/judge_*.yaml`
*   **`SweAgentJudge`**:
    *   Leverages the SWE-agent framework for evaluation tasks.
    *   Can perform complex evaluations by running the agent in a Docker environment.
    *   Configuration involves SWE-agent specific parameters in its YAML file and requires Docker.
    *   Configuration files for the Agent-Planner can be found in `config/sweagent/judge_*.yaml`

Both planners and judges are registered and can be selected via the CLI using their respective names (e.g., `simple_lm`, `sweagent`).

## Usage Examples

The primary way to interact with AblationBench is through its command-line interface.
Use `ablation-bench --help` to get all the different parameters that can be set as part of each interface.

### 1. Generating Ablation Plans

Use the `ablation-bench plan` command:

```bash
ablation-bench plan \
    --planner [planner_type] \
    --planner-config [path/to/your_planner_config.yaml] \
    --model-name "your_chosen_lm_identifier" \
    --dataset [huggingface_dataset_name] \
    --split [dataset_split] \
    --num-ablations [number_of_ablations_to_generate] \
    --parallelism [number_of_parallel_workers] \
    --output-dir [path/to/output_directory]
```

*   `planner_type`: `simple_lm` or `sweagent`.
*   `planner_config`: Path to the YAML configuration for the chosen planner.
*   `model_name`: Identifier for the LM to be used by the planner (e.g., "openai/gpt-4o", "anthropic/claude-3-opus-20240229"). See [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for supported models.
*   `dataset`: The Hugging Face dataset to use.
*   `split`: Dataset split (e.g., "dev", "test").
*   `num_ablations` (optional): Target number of ablations to generate per paper (default: 5).
*   `output_dir` (optional): Directory where generated plans and logs will be saved (summary in `plans.json` and individual `*.jsonl` files per paper) (default: `./runs/<datetime>`).

### 2. Evaluating Ablation Plans

Use the `ablation-bench eval` command:

```bash
ablation-bench eval \
    --judge [judge_type] \
    --judge-config [path/to/your_judge_config.yaml] \
    --model-name "your_chosen_lm_for_judging" \
    --dataset [huggingface_dataset_name] \
    --split [dataset_split] \
    --generated-plans-path [path/to/directory_with_generated_plans] \
    --top-k [N] \
    --parallelism [number_of_parallel_workers] \
    --output-dir [path/to/evaluation_output_directory]
```

*   `judge_type`: `simple_lm` or `sweagent`.
*   `judge_config`: Path to the YAML configuration for the chosen judge.
*   `model_name`: Identifier for the LM to be used by the judge.
*   `dataset`: The Hugging Face dataset to use.
*   `generated_plans_path`: Path to the directory containing the ablation plans generated by the `plan` command.
*   `top_k` (optional): Evaluate only the top K suggestions from each plan.
*   `output_dir` (optional): Directory where evaluation results (`evaluations.json` containing precision, recall, F1 scores per paper) will be saved (default: `./runs/<datetime>`).

### 3. Evaluating Judge Performance

Use the `ablation-bench eval-judge` command:

```bash
ablation-bench eval-judge \
    --dataset [huggingface_dataset_for_judge_eval] \
    --judge-evaluations-path [path/to/judge_outputs_being_evaluated]
```

*   `dataset`: The Hugging Face dataset specifically designed for evaluating judges (e.g., "ai-coscientis/researcher-ablation-judge-eval").
*   `judge_evaluations_path`: Path to the directory containing the outputs generated by a judge run (the files that `eval-judge` will score against its ground truth).

This command will output metrics like precision, recall, and F1 score for the judge's performance.
