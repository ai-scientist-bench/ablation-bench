"""Defines the core data types and Pydantic models used throughout the ablations-bench project."""
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


class AblationSuggestion(BaseModel):
    """Model representing an ablation suggestion.

    Attributes:
        name: The name of the ablation.
        ablated_part: The part of the model or system being ablated.
        action: The action to perform for the ablation (REPLACE, REMOVE, ADD).
        replacement: The replacement value or configuration, if applicable.
        metrics: A list of metrics expected to be affected by this ablation.
    """

    name: str
    ablated_part: str
    action: Literal["REPLACE", "REMOVE", "ADD"]
    replacement: list[Any] | dict[str, Any] | None = None
    metrics: list[str]


class AblationSuggestionPred(BaseModel):
    """Model representing a prediction for an ablation suggestion\'s relevance.

    This is typically used to assess if a suggested ablation was part of an original plan
    or mentioned in a review.

    Attributes:
        name_in_paper: The name of the ablation as it appears in the paper/source.
        name_in_plan: The name of the ablation as it appears in a generated plan (if any).
    """

    name_in_paper: str
    name_in_plan: list[str] | str | None = None

    @property
    def label(self) -> bool:
        """Calculate the score based on the name in the paper and plan."""
        return self.name_in_plan is not None


class PredResponse(BaseModel):
    """Base model for a response that includes predictions and discussion.

    Attributes:
        pred_class_type: A ClassVar indicating the Pydantic model type for the predictions.
        discussion: Textual discussion or reasoning accompanying the predictions.
        cost: The cost associated with generating this response (e.g., LLM call cost).
    """
    pred_class_type: ClassVar[type[BaseModel]]
    discussion: str
    cost: float

    @classmethod
    def from_lm_response(
        cls, response: str, cost: float, post_process_prediction: Callable[[list], list] | None = None
    ) -> "PredResponse":
        """Parse the LM response into an PredResponse."""
        discussion_match = re.search(r"<discussion>(.*?)</discussion>", response, re.DOTALL)
        discussion = discussion_match.group(1).strip() if discussion_match else ""
        predictions_match = re.search(r"<predictions>(.*?)</predictions>", response, re.DOTALL)
        predictions = (
            [pred for pred in predictions_match.group(1).strip().split("\n") if pred] if predictions_match else []
        )
        if post_process_prediction is not None:
            predictions = post_process_prediction(predictions)
        predictions = [cls.pred_class_type(**json.loads(pred)) for pred in predictions]
        return cls(discussion=discussion, predictions=predictions, cost=cost)


class AblationSuggestionPredResponse(PredResponse):
    """A response containing a list of ablation suggestion predictions.

    Attributes:
        predictions: A list of ablation suggestion predictions.
        pred_class_type: Specifies that predictions are of type AblationSuggestionPred.
    """

    predictions: list[AblationSuggestionPred]
    pred_class_type: ClassVar[type[AblationSuggestionPred]] = AblationSuggestionPred


class MissingAblationSuggestionPred(BaseModel):
    """Model representing a prediction about a missing ablation suggestion.

    This is used to identify ablations that were in a plan but perhaps not
    mentioned or found elsewhere (e.g., in a review).

    Attributes:
        name_in_plan: The name of the ablation as it appears in the plan.
        appears_in_review: Whether this ablation suggestion also appears in a review.
    """

    name_in_plan: str
    appears_in_review: bool = False


class MissingAblationSuggestionPredResponse(PredResponse):
    """A response containing a list of predictions about missing ablation suggestions.

    Attributes:
        predictions: A list of missing ablation suggestion predictions.
        pred_class_type: Specifies that predictions are of type MissingAblationSuggestionPred.
    """

    predictions: list[MissingAblationSuggestionPred]
    pred_class_type: ClassVar[type[MissingAblationSuggestionPred]] = MissingAblationSuggestionPred


class AblationPlanPredResponse(PredResponse):
    """A response containing a list of ablation plan (suggestion) predictions.

    Attributes:
        predictions: A list of ablation suggestions.
        pred_class_type: Specifies that predictions are of type AblationSuggestion.
    """

    predictions: list[AblationSuggestion]
    pred_class_type: ClassVar[type[AblationSuggestion]] = AblationSuggestion


class ModelConfig(BaseModel):
    """Configuration for a language model.

    Attributes:
        name: The name or identifier of the model.
        temperature: The sampling temperature for the model.
        top_p: The nucleus sampling (top-p) parameter for the model.
    """
    name: str
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    reasoning_effort: Literal["low", "medium", "high"] | None = None


class SimpleLMConfig(BaseModel):
    """Configuration for a simple Language Model-based judge or planner.

    Attributes:
        model: The language model configuration.
        prompts: A dictionary of prompts to be used, keyed by prompt name.
        output_dir: The directory where outputs will be saved.
        parallelism: The number of parallel requests to make to the LM.
    """

    model: ModelConfig
    prompts: dict[str, str]
    output_dir: Path
    parallelism: int = Field(1, ge=1)


class AblationPlanSimpleLMConfig(SimpleLMConfig):
    """Configuration for a SimpleLM planner specifically for ablation plans.

    Attributes:
        num_ablations: The target number of ablations to generate in a plan.
    """
    num_ablations: int = Field(5, ge=1)

class MajorityJudgeConfig(BaseModel):
    """Configuration for the Majority judge."""

    model: ModelConfig
    judge_output_dirs: list[str] = Field(..., description="List of paths to judge output directories")
    output_dir: Path
    parallelism: int = Field(1, ge=1)
class SingleResult(BaseModel):
    """Represents a single numerical result, potentially with standard deviation.

    Attributes:
        result: The numerical value of the result.
        std_dev: The standard deviation of the result, if applicable.
    """
    result: float
    std_dev: float = 0.0


class EvaluationResult(BaseModel):
    """Results from an evaluation run.

    Attributes:
        precision: The precision score.
        recall: The recall score.
        f1_score: The F1 score.
        cost: The total cost associated with the evaluation.
    """

    precision: SingleResult
    recall: SingleResult
    f1_score: SingleResult
    ndcg_score: SingleResult | None = None
    cost: float = 0.0


class DatasetForEvaluation(str, Enum):
    """Enumeration of dataset identifiers for evaluation purposes.

    These typically point to Hugging Face Hub dataset names.

    Attributes:
        ReviewerAssist: Dataset for the ReviewerAssist benchmark.
        ResearcherAssist: Dataset for the ResearcherAssist benchmark.
    """

    ReviewerAssist = "ai-coscientist/reviewer-ablation-bench"
    ResearcherAssist = "ai-coscientist/researcher-ablation-bench"


class DatasetSplit(str, Enum):
    dev = "dev"
    test = "test"


class DatasetForJudgeEvaluation(str, Enum):
    """Enumeration of dataset identifiers specifically for judge evaluation.

    These may differ from general evaluation datasets if judges are evaluated
    on specific subsets or different data structures.

    Attributes:
        ReviewerAssist: Dataset for judge evaluation in ReviewerAssist mode.
        ResearcherAssist: Dataset for judge evaluation in ResearcherAssist mode.
    """

    ReviewerAssist = "ai-coscientist/reviewer-ablation-judge-eval"
    ResearcherAssist = "ai-coscientist/researcher-ablation-judge-eval"


PredictedField: dict[DatasetForJudgeEvaluation, str] = {
    DatasetForJudgeEvaluation.ReviewerAssist: "appears_in_review",
    DatasetForJudgeEvaluation.ResearcherAssist: "name_in_plan",
}


NonPredictedField: dict[DatasetForJudgeEvaluation, str] = {
    DatasetForJudgeEvaluation.ReviewerAssist: "name_in_plan",
    DatasetForJudgeEvaluation.ResearcherAssist: "name_in_paper",
}

