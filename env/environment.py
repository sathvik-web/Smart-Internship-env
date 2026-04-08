from __future__ import annotations

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

    def reset(self) -> Observation:
        self.current_index = 0
        self.last_reward = 0.0
        return self._task_to_observation(self.tasks[self.current_index])

    def step(self, action):
        task = self.tasks[self.current_index]
        progress_ratio = self.current_index / len(self.tasks)

        reward_obj = compute_reward(action, task, progress_ratio)
        reward = float(reward_obj.total)

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