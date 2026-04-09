def easy_grader(action):
    """
    Task: Decide whether to apply or ignore
    """
    decision = action.get("decision", "")

    if decision == "apply":
        return 1.0   # correct decision
    elif decision == "ignore":
        return 0.5   # partially okay
    else:
        return 0.0


def medium_grader(action):
    """
    Task: Evaluate relevance score (0–1)
    """
    score = action.get("relevance_score", 0)

    # Ideal score ~0.8 for good match
    if score >= 0.7:
        return 1.0
    elif score >= 0.4:
        return 0.6
    else:
        return 0.2


def hard_grader(action):
    """
    Task: Rank internships correctly
    """
    ranking = action.get("ranking", [])

    correct_order = ["Backend Intern", "ML Intern"]

    if ranking == correct_order:
        return 1.0
    elif len(ranking) > 0 and ranking[0] == "Backend Intern":
        return 0.6   # partially correct
    else:
        return 0.2