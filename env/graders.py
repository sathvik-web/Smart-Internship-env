# env/graders.py

def easy_grader(state):
    return float(state.get("last_reward", 0.0))


def medium_grader(state):
    return float(state.get("last_reward", 0.0))


def hard_grader(state):
    return float(state.get("last_reward", 0.0))