def easy_grader(action: dict) -> float:
    """
    Task: Decide whether to apply or ignore.
    Expected: decision == "apply"
    """
    decision = action.get("decision", "")

    if decision == "apply":
        return 1.0
    elif decision == "ignore":
        return 0.5
    else:
        return 0.0


def medium_grader(action: dict) -> float:
    """
    Task: Evaluate relevance score (0–1).
    Ideal: relevance_score >= 0.7 for a good skill match.
    """
    score = action.get("relevance_score", 0.0)

    if score >= 0.7:
        return 1.0
    elif score >= 0.4:
        return 0.6
    else:
        return 0.2


def hard_grader(action: dict) -> float:
    """
    Task: Rank internships correctly.
    Correct order (matches tasks.py correct_ranking): ["ML Intern", "Backend Intern"]
    """
    ranking = action.get("ranking", [])

    # Must match tasks.py  correct_ranking = ["ML Intern", "Backend Intern"]
    correct_order = ["ML Intern", "Backend Intern"]

    if ranking == correct_order:
        return 1.0
    elif len(ranking) > 0 and ranking[0] == "ML Intern":
        return 0.6   # first item correct, partial credit
    else:
        return 0.2