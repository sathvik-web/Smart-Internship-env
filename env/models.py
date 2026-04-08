from __future__ import annotations
from pydantic import ConfigDict
from typing import Literal, Callable, Optional
from pydantic import BaseModel, Field, field_validator

class InternshipOption(BaseModel):
    internship_title: str
    description: str
    required_skills: list[str] = Field(default_factory=list)


class Observation(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    internship_title: str
    description: str
    required_skills: list[str]
    student_skills: list[str]
    internship_options: list[InternshipOption] = Field(default_factory=list)


class Action(BaseModel):
    decision: Literal["apply", "ignore"]
    relevance_score: float = Field(ge=0.0, le=1.0)
    ranking: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=3)

    @field_validator("ranking")
    @classmethod
    def ranking_titles_must_be_unique(cls, value: list[str]) -> list[str]:
        lowered = [item.strip().lower() for item in value]
        if len(lowered) != len(set(lowered)):
            raise ValueError("ranking entries must be unique")
        return value


class Reward(BaseModel):
    total: float = Field(ge=0.0, le=1.0)
    decision_score: float = Field(ge=0.0, le=1.0)
    score_closeness: float = Field(ge=0.0, le=1.0)
    reasoning_score: float = Field(ge=0.0, le=1.0)
    ranking_score: float = Field(ge=0.0, le=1.0)
    progress_bonus: float = Field(ge=0.0, le=0.1)
    penalty: float = Field(ge=0.0, le=1.0)
    feedback: str


class InternshipTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    internship_title: str
    description: str
    required_skills: list[str]
    student_skills: list[str]
    correct_decision: Literal["apply", "ignore"]
    true_score: float = Field(ge=0.0, le=1.0)
    expected_reasoning_keywords: list[str] = Field(default_factory=list)
    internship_options: list[InternshipOption] = Field(default_factory=list)
    correct_ranking: list[str] = Field(default_factory=list)

    # REQUIRED FOR VALIDATOR
    grader: Callable