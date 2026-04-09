"""
SmartInternshipEnv baseline inference script.
Produces strict [START] / [STEP] / [END] logs.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

from env.environment import InternshipEnv
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

BENCHMARK = "SmartInternshipEnv"
MAX_STEPS = 16


def _heuristic_action(obs) -> Action:
    required = {s.lower() for s in obs.required_skills}
    student = {s.lower() for s in obs.student_skills}

    overlap = len(required & student)
    coverage = overlap / len(required) if required else 1.0

    ranking = []
    if obs.internship_options:
        scored = []
        for option in obs.internship_options:
            req = {s.lower() for s in option.required_skills}
            cov = len(req & student) / len(req) if req else 1.0
            scored.append((option.internship_title, cov))
        scored.sort(key=lambda x: (-x[1], x[0]))
        ranking = [x[0] for x in scored]
        coverage = scored[0][1]

    return Action(
        decision="apply" if coverage >= 0.45 else "ignore",
        relevance_score=round(max(0.0, min(1.0, coverage)), 3),
        ranking=ranking,
        reasoning="Skill overlap based decision",
    )


def _model_action(client: OpenAI, obs) -> Action:
    prompt = (
        "Return ONLY valid JSON with fields: decision, relevance_score, ranking, reasoning.\n"
        f"Task: {obs.model_dump_json()}"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt},
        ],
    )

    raw = (response.choices[0].message.content or "{}").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    first = raw.find("{")
    last = raw.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found")

    return Action.model_validate(json.loads(raw[first:last + 1]))


def run_task(env: InternshipEnv, client: OpenAI | None, task_id: str, llm_enabled: bool) -> bool:
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        observation = env.reset(task=task_id)
    except Exception:
        print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
        return llm_enabled

    rewards: list[float] = []
    success = False
    steps_taken = 0
    done = False

    try:
        for step in range(1, MAX_STEPS + 1):
            if done or observation is None:
                break

            error = None
            try:
                action = _model_action(client, observation) if (llm_enabled and client is not None) else _heuristic_action(observation)
            except Exception:
                action = _heuristic_action(observation)
                llm_enabled = False
                error = None

            current_task_id = observation.task_id

            try:
                observation, reward, done, _info = env.step(action)
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            reward = float(max(0.0, min(1.0, reward)))
            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            err_str = "null" if error is None else error
            print(
                f"[STEP] step={step} task={current_task_id} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error={err_str}",
                flush=True,
            )

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= 0.1
    except Exception:
        score = 0.0
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps_taken} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return llm_enabled


def main() -> int:
    client = None
    llm_enabled = False
    if API_KEY:
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
            try:
                client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                )
                llm_enabled = True
            except Exception:
                client = None
                llm_enabled = False
        except Exception:
            client = None
            llm_enabled = False
    else:
        print(
            "[WARN] No API key found (HF_TOKEN, OPENAI_API_KEY, API_KEY). Using heuristic mode.",
            flush=True,
        )

    env = InternshipEnv()
    for task_id in ["easy-apply-ignore-001", "medium-relevance-001", "hard-ranking-001"]:
        llm_enabled = run_task(env, client, task_id, llm_enabled)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
