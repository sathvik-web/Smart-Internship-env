from __future__ import annotations

from env.tasks import load_tasks


class InternshipEnv:
    def __init__(self):
        self.tasks = load_tasks()
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self.tasks[self.current_index]

    def step(self, action):
        task = self.tasks[self.current_index]

        progress_ratio = self.current_index / len(self.tasks)

        if not callable(task.grader):
            raise ValueError("Grader missing")

        # ✅ RETURNS FLOAT
        reward = task.grader(action, task, progress_ratio)

        self.current_index += 1
        done = self.current_index >= len(self.tasks)

        next_obs = None if done else self.tasks[self.current_index]

        return next_obs, reward, done, {}

    def state(self):
        return {"current_index": self.current_index}