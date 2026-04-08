from __future__ import annotations

from env.models import InternshipOption, InternshipTask
from env.reward import compute_reward


# ✅ SIMPLE WRAPPER GRADERS (VERY IMPORTANT)
def easy_grader(action, task, progress):
    return compute_reward(action, task, progress).total


def medium_grader(action, task, progress):
    return compute_reward(action, task, progress).total


def hard_grader(action, task, progress):
    return compute_reward(action, task, progress).total


def load_tasks() -> list[InternshipTask]:

    easy_task = InternshipTask(
        task_id="easy-apply-ignore-001",
        difficulty="easy",
        objective="Classify whether the student should apply.",
        internship_title="Backend Engineering Intern",
        description="Build APIs",
        required_skills=["python", "sql"],
        student_skills=["python", "sql"],
        correct_decision="apply",
        true_score=0.74,
        expected_reasoning_keywords=["python", "backend"],
        grader=easy_grader,   # ✅ FIXED
    )

    medium_task = InternshipTask(
        task_id="medium-relevance-001",
        difficulty="medium",
        objective="Provide relevance score",
        internship_title="ML Intern",
        description="ML pipelines",
        required_skills=["python", "mlops", "docker"],
        student_skills=["python", "docker"],
        correct_decision="apply",
        true_score=0.61,
        expected_reasoning_keywords=["mlops", "docker"],
        grader=medium_grader,   # ✅ FIXED
    )

    hard_task = InternshipTask(
        task_id="hard-ranking-001",
        difficulty="hard",
        objective="Rank internships",
        internship_title="Multi-role",
        description="Ranking",
        required_skills=["ranking"],
        student_skills=["python"],
        correct_decision="apply",
        true_score=0.81,
        expected_reasoning_keywords=["ranking"],
        internship_options=[
            InternshipOption(
                internship_title="ML Intern",
                description="ML",
                required_skills=["python"],
            ),
            InternshipOption(
                internship_title="Backend Intern",
                description="API",
                required_skills=["fastapi"],
            ),
        ],
        correct_ranking=["ML Intern", "Backend Intern"],
        grader=hard_grader,   # ✅ FIXED
    )

    return [easy_task, medium_task, hard_task]