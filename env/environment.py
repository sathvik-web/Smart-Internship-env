from __future__ import annotations
from env.graders import easy_grader, medium_grader, hard_grader
from env.models import Observation
from env.tasks import load_tasks
from env.reward import compute_reward


class InternshipEnv:
    def __init__(self):
        self.tasks = load_tasks()
        self.current_index = 0
        self.last_reward = 0.0

    def _task_to_observation(self, task) -> Observation:
        return Observation(
            task_id=task.task_id,
            difficulty=task.difficulty,
            objective=task.objective,
            internship_title=task.internship_title,
            description=task.description,
            required_skills=task.required_skills,
            student_skills=task.student_skills,
            internship_options=task.internship_options,
        )

    # ✅ RESET
    def reset(self, task: str = None) -> Observation:
        self.last_reward = 0.0

        if task is not None:
            found = False
            for i, t in enumerate(self.tasks):
                if t.task_id == task:
                    self.current_index = i
                    found = True
                    break
            if not found:
                raise ValueError(f"Task {task} not found")
        else:
            self.current_index = 0

        return self._task_to_observation(self.tasks[self.current_index])

    # ✅ UPDATED STEP (IMPORTANT FIX)
    def step(self, action):
        task = self.tasks[self.current_index]
        progress_ratio = self.current_index / len(self.tasks)

        # 🔹 Base reward from graders (MAIN LOGIC)
        if task.difficulty == "easy":
            base_reward = easy_grader(action)

        elif task.difficulty == "medium":
            base_reward = medium_grader(action)

        elif task.difficulty == "hard":
            base_reward = hard_grader(action)

        else:
            base_reward = 0.0

        # 🔹 Bonus shaping (optional but good)
        reward_obj = compute_reward(action, task, progress_ratio)
        shaping_bonus = float(reward_obj.total) * 0.2   # small influence

        # 🔹 Final reward (weighted)
        reward = min(1.0, base_reward * 0.8 + shaping_bonus)

        self.last_reward = reward

        self.current_index += 1
        done = self.current_index >= len(self.tasks)

        next_obs = None if done else self._task_to_observation(self.tasks[self.current_index])

        return next_obs, reward, done, {}

    def state(self):
        return {
            "current_index": self.current_index,
            "last_reward": self.last_reward
        }