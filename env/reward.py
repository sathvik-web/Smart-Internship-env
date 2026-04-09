from __future__ import annotations

import re

from env.models import Action, InternshipTask, Reward


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _reasoning_keyword_score(reasoning: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0

    normalized_reasoning = re.sub(r"\s+", " ", reasoning.lower()).strip()
    hits = sum(1 for keyword in keywords if keyword.lower() in normalized_reasoning)
    return _clamp(hits / len(keywords))


def _ranking_score(predicted_ranking: list[str], expected_ranking: list[str]) -> float:
    if not expected_ranking:
        return 1.0

    if not predicted_ranking:
        return 0.0

    expected_positions = {title.lower(): idx for idx, title in enumerate(expected_ranking)}
    predicted_positions = {title.lower(): idx for idx, title in enumerate(predicted_ranking)}

    pairwise_total = 0
    pairwise_correct = 0

    expected_items = [item.lower() for item in expected_ranking]
    for i in range(len(expected_items)):
        for j in range(i + 1, len(expected_items)):
            a = expected_items[i]
            b = expected_items[j]
            pairwise_total += 1
            if a not in predicted_positions or b not in predicted_positions:
                continue
            if predicted_positions[a] < predicted_positions[b]:
                pairwise_correct += 1

    if pairwise_total == 0:
        return 0.0

    return _clamp(pairwise_correct / pairwise_total)


def compute_reward(action: Action, task: InternshipTask, progress_ratio: float) -> Reward:
    decision = action.decision
    relevance_score = action.relevance_score
    reasoning = action.reasoning
    ranking = action.ranking

    # ✅ DEFINE ALL COMPONENTS (THIS WAS MISSING)
    decision_score = 1.0 if decision == task.correct_decision else 0.0
    score_closeness = _clamp(1.0 - abs(relevance_score - task.true_score))
    reasoning_score = _reasoning_keyword_score(reasoning, task.expected_reasoning_keywords)
    ranking_score = _ranking_score(ranking, task.correct_ranking)

    is_hard = task.difficulty == "hard"
    if is_hard:
        weighted_core = (
            0.30 * decision_score
            + 0.25 * score_closeness
            + 0.20 * reasoning_score
            + 0.25 * ranking_score
        )
    else:
        weighted_core = (
            0.45 * decision_score
            + 0.35 * score_closeness
            + 0.20 * reasoning_score
        )

    progress_bonus = 0.08 * _clamp(progress_ratio)

    penalty = 0.0
    if decision != task.correct_decision:
        confidence_gap = abs(relevance_score - 0.5)
        penalty += 0.10 + 0.10 * confidence_gap

    score_error = abs(relevance_score - task.true_score)
    if score_error > 0.45:
        penalty += 0.12
    elif score_error > 0.30:
        penalty += 0.06

    if reasoning_score < 0.20:
        penalty += 0.08
    elif reasoning_score < 0.40:
        penalty += 0.04

    if task.correct_ranking and not ranking:
        penalty += 0.08

    total = _clamp(weighted_core + progress_bonus - penalty)

    feedback = (
        f"decision={decision_score:.2f}, score={score_closeness:.2f}, "
        f"reasoning={reasoning_score:.2f}, ranking={ranking_score:.2f}, "
        f"bonus={progress_bonus:.2f}, penalty={penalty:.2f}"
    )

    return Reward(
        total=total,
        decision_score=decision_score,
        score_closeness=score_closeness,
        reasoning_score=reasoning_score,
        ranking_score=ranking_score,
        progress_bonus=progress_bonus,
        penalty=_clamp(penalty),
        feedback=feedback,
    )