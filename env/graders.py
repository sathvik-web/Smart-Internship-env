from env.reward import compute_reward


def easy_grader(action, task, progress) -> float:
    return compute_reward(action, task, progress).total


def medium_grader(action, task, progress) -> float:
    return compute_reward(action, task, progress).total


def hard_grader(action, task, progress) -> float:
    return compute_reward(action, task, progress).total