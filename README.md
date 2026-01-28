# AblationBench: Evaluating Language Models on Ablation Planning in Empirical AI Research

AblationBench is a comprehensive benchmarking tool designed to evaluate and facilitate the creation of ablation plans, particularly in the context of AI and Machine Learning research. It provides a standardized framework for generating ablation plans and evaluating their quality using different LM-based judges, including chain-of-thought (CoT) prompting and using [SWE-agent](https://github.com/SWE-agent/SWE-agent) based judge.


## 🛠️ Installation

1.  **Prerequisites**:
    *   Python 3.11 or higher.
    *   Docker (if using SWE-agent based planners or judges).

2.  **Create and activate a virtual environment (recommended)**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    # On Windows: .venv\Scripts\activate
    ```

3.  **Install the package and its dependencies**:
    ```bash
    pip install -e .
    ```
    This installs the package in editable mode. If you prefer a standard installation, use `pip install .`.

4.  **(Optional) Install SWE-agent**:

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

    You should build docker images using the provided script in docker/ folder, in order to be able to use the SWE-agent planner and judge:
    
    ```bash
    cd docker/ && ./build_base_images.sh
    ```

## 🤗 Datasets

The benchmark utilizes the following datasets:

*   `data/ai-coscientist/author-ablation`: For ablation planning and evaluation given paper's method section (author-assist mode).
*   `data/ai-coscientist/reviewer-ablation`: For tasks related to identifying missing ablations given a full paper (reviewer-assist mode).
*   `data/ai-coscientist/author-eval`: For evaluating the performance of different judges for the AuthorAblation.
*   `data/ai-coscientist/reviewer-eval`: For evaluating the performance of different judges for the ReviewerAblation.


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


### 👩‍🔬 AuthorAblation

Here's a quick start example of how to generate an ablation plan using the LM-Planner, then evaluate it using the LMJudge:


1.  **Run the planning command**:
    ```bash
    ablation-bench plan \
        --planner simple_lm \
        --planner-config config/simple_lm/plan_ablations.yaml \
        --model-name "openrouter/openai/o3-mini-high" \
        --dataset "data/ai-coscientist/author-ablation" \
        --split "test" \
        --num-ablations 5 \
        --output-dir "./runs/plan/author/simple_lm/o3-mini-high"
    ```
    This will generate ablation plans for all the papers in the test set of AuthorAblation. Each plan will contain up to 5 generated ablation studies using o3-mini-high. The command will save these to the `./runs/plan/author/simple_lm/o3-mini-high` directory.

2.  **Run the evaluation command**:
    ```bash
    # Run the evaluation on all of the judges separately
    for judge_model in "openai/gpt-4o" "anthropic/claude-3.5-sonnet" "openai/o3-mini-high"
    do
        ablation-bench eval \
            --judge simple_lm \
            --judge-config config/simple_lm/judge_ablation_suggestions.yaml \
            --model-name "openrouter/${judge_model}" \
            --dataset "data/ai-coscientist/author-ablation" \
            --split "test" \
            --generated-plans-path "./runs/author/plan/simple_lm/o3-mini-high" \
            --parallelism 5 \
            --output-dir "./runs/eval/author/simple_lm/o3-mini-high/${judge_model}"
    done

    # Run the majority voting to get the final result
    ablation-bench eval \
        --judge majority_judge \
        --judge-config config/majority_judge/majority_vote_author.yaml \
        --model-name "openrouter/dummy" \
        --dataset "data/ai-coscientist/author-ablation" \
        --split "test" \
        --generated-plans-path "./runs/author/plan/simple_lm/o3-mini-high" \
        --parallelism 5 \
        --output-dir "./runs/eval/author/simple_lm/o3-mini-high/majority_vote"
    ```
    This will evaluate ablation plans generated in the previous step, for all the papers in the test set of AuthorAblation. The command will save evaluation results to the `./runs/eval/author/simple_lm/o3-mini-high` directory, and print them to the screen as well.

### 🔍 ReviewerAblation

Here's a quick start example of how to generate a missing ablation plan using the LM-Planner, then evaluate it using the LMJudge:

1.  **Run the planning command**:
    ```bash
    ablation-bench plan \
        --planner simple_lm \
        --planner-config config/simple_lm/plan_missing_ablations.yaml \
        --model-name "openrouter/openai/o3-mini-high" \
        --dataset "data/ai-coscientist/reviewer-ablation" \
        --split "test" \
        --num-ablations 2 \
        --output-dir "./runs/plan/reviewer/simple_lm/o3-mini-high"
    ```
    This will generate ablation plans for all the papers in the test set of ReviewerAblation. Each plan will contain up to 2 generated ablation studies using o3-mini-high. The command will save these to the `./runs/plan/reviewer/simple_lm/o3-mini-high` directory.

2.  **Run the evaluation command**:
    ```bash
    # Run the evaluation on all of the judges separately
    for judge_model in "openai/gpt-4o" "anthropic/claude-3.5-sonnet" "openai/o3-mini-high"
    do
        ablation-bench eval \
            --judge simple_lm \
            --judge-config config/simple_lm/judge_missing_ablation_suggestions.yaml \
            --model-name "openrouter/${judge_model}$" \
            --dataset "data/ai-coscientist/reviewer-ablation" \
            --split "test" \
            --generated-plans-path "./runs/plan/reviewer/simple_lm/o3-mini-high" \
            --parallelism 5 \
            --output-dir "./runs/eval/reviewer/simple_lm/o3-mini-high/${judge_model}"
    done
    

    # Run the majority voting to get the final result
    ablation-bench eval \
        --judge majority_judge \
        --judge-config config/majority_judge/majority_vote_reviewer.yaml \
        --model-name "openrouter/dummy$" \
        --dataset "data/ai-coscientist/reviewer-ablation" \
        --split "test" \
        --generated-plans-path "./runs/plan/reviewer/simple_lm/o3-mini-high" \
        --parallelism 5 \
        --output-dir "./runs/eval/reviewer/simple_lm/o3-mini-high/majority_vote"
    ```
    This will evaluate ablation plans generated in the previous step, for all the papers in the test set of ReviewerAblation. The command will save evaluation results to the `./runs/eval/reviewer/simple_lm/o3-mini-high` directory, and print them to the screen as well.


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
*   ** MajorityJudge **:
    *   An ensemble of models for more reliable evaluation. 
    *   Used mainly to reduce intra-model (also called self-model) bias.
    *   Configuration files for the MajorityJudge can be found in `config/majority_judge/*.yaml`.

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
    --dataset [dataset_path] \
    --split [dataset_split] \
    --num-ablations [number_of_ablations_to_generate] \
    --parallelism [number_of_parallel_workers] \
    --output-dir [path/to/output_directory]
```

*   `planner_type`: `simple_lm` or `sweagent`.
*   `planner_config`: Path to the YAML configuration for the chosen planner.
*   `model_name`: Identifier for the LM to be used by the planner (e.g., "openai/gpt-4o", "anthropic/claude-3-opus-20240229"). See [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for supported models.
*   `dataset`: The path to dataset to use.
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
    --dataset [dataset_path] \
    --split [dataset_split] \
    --generated-plans-path [path/to/directory_with_generated_plans] \
    --top-k [N] \
    --parallelism [number_of_parallel_workers] \
    --output-dir [path/to/evaluation_output_directory]
```

*   `judge_type`: `simple_lm`, `sweagent` or `majority_judge`.
*   `judge_config`: Path to the YAML configuration for the chosen judge.
*   `model_name`: Identifier for the LM to be used by the judge.
*   `dataset`: The local path of dataset to use.
*   `generated_plans_path`: Path to the directory containing the ablation plans generated by the `plan` command.
*   `top_k` (optional): Evaluate only the top K suggestions from each plan.
*   `output_dir` (optional): Directory where evaluation results (`evaluations.json` containing precision, recall, F1 scores per paper) will be saved (default: `./runs/<datetime>`).

### 3. Evaluating Judge Performance

Use the `ablation-bench eval-judge` command:

```bash
ablation-bench eval-judge \
    --dataset [dataset_path_for_judge_eval] \
    --judge-evaluations-path [path/to/judge_outputs_being_evaluated]
```

*   `dataset`: The dataset path specifically designed for evaluating judges (e.g., "data/ai-coscientis/author-eval").
*   `judge_evaluations_path`: Path to the directory containing the outputs generated by a judge run (the files that `eval-judge` will score against its ground truth).

This command will output metrics like precision, recall, and F1 score for the judge's performance.
