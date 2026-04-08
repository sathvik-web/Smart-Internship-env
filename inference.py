import os
from env.environment import InternshipEnv

MODEL_NAME = os.getenv("MODEL_NAME", "baseline-model")


def run_task(env, task_id):
    print(f"[START] task={task_id} env=SmartInternshipEnv model={MODEL_NAME}")

    obs = env.reset()
    done = False
    step_count = 0
    rewards = []

    while not done:
        step_count += 1

        action = {
            "decision": "apply",
            "relevance_score": 0.5,
            "ranking": ["ML Intern", "Backend Intern"],
            "reasoning": "python backend ml"
        }

        obs, reward, done, _ = env.step(action)
        rewards.append(reward)

        print(f"[STEP] step={step_count} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

    score = rewards[-1] if rewards else 0.0
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])

    print(f"[END] success={str(score>0).lower()} steps={step_count} score={score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    env = InternshipEnv()

    tasks = [
        "easy-apply-ignore-001",
        "medium-relevance-001",
        "hard-ranking-001"
    ]

    for task in tasks:
        run_task(env, task)