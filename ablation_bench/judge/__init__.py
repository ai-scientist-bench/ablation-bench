"""Initializes the judge module, providing access to different judge implementations."""
from enum import Enum

from .abstract import JUDGE_REGISTRY, Judge
from .simple_lm import SimpleLMJudge
from .sweagent import SweAgentJudge

JudgeType = Enum("JudgeType", {key: key for key in JUDGE_REGISTRY.keys()})

__all__ = [Judge, SimpleLMJudge, SweAgentJudge, JudgeType]
