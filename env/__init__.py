# env/__init__.py
from env.graders import easy_grader, medium_grader, hard_grader

__all__ = ["easy_grader", "medium_grader", "hard_grader"]

def _lazy_load():
    from env.environment import InternshipEnv
    from env.models import Action, Observation, Reward
    return InternshipEnv, Action, Observation, Reward