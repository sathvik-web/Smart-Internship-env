from __future__ import annotations

from typing import Any

from env.models import Action, Observation
from env.reward import compute_reward
from env.tasks import load_tasks


class InternshipEnv:
	"""OpenEnv-style internship recommendation environment."""

	def __init__(self) -> None:
		self._tasks = load_tasks()
		self._index = 0
		self._done = False
		self._episode_rewards: list[float] = []
		self._last_observation: Observation | None = None

	def _build_observation(self, index: int) -> Observation:
		task = self._tasks[index]
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
		if not self._tasks:
			raise RuntimeError("No tasks are available in the environment")

		self._index = 0
		self._done = False
		self._episode_rewards = []
		self._last_observation = self._build_observation(self._index)
		return self._last_observation

	def state(self) -> dict[str, Any]:
		return {
			"current_index": self._index,
			"total_tasks": len(self._tasks),
			"done": self._done,
			"episode_rewards": list(self._episode_rewards),
			"average_reward": (
				sum(self._episode_rewards) / len(self._episode_rewards) if self._episode_rewards else 0.0
			),
			"current_task_id": None if self._done else self._tasks[self._index].task_id,
		}

	def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
		if self._done:
			if self._last_observation is None:
				raise RuntimeError("Environment not initialized. Call reset() before step().")
			return self._last_observation, 0.0, True, {"error": "episode already finished"}

		task = self._tasks[self._index]
		progress_ratio = (self._index + 1) / len(self._tasks)
		reward = compute_reward(action=action, task=task, progress_ratio=progress_ratio)
		self._episode_rewards.append(reward.total)

		info: dict[str, Any] = {
			"task_id": task.task_id,
			"difficulty": task.difficulty,
			"objective": task.objective,
			"reward_breakdown": reward.model_dump(),
			"ground_truth": {
				"correct_decision": task.correct_decision,
				"true_score": task.true_score,
				"correct_ranking": task.correct_ranking,
			},
		}

		self._index += 1
		self._done = self._index >= len(self._tasks)

		if self._done:
			if self._last_observation is None:
				self._last_observation = self._build_observation(len(self._tasks) - 1)
			info["episode_summary"] = {
				"steps": len(self._episode_rewards),
				"mean_reward": sum(self._episode_rewards) / len(self._episode_rewards),
			}
			return self._last_observation, reward.total, True, info

		self._last_observation = self._build_observation(self._index)
		return self._last_observation, reward.total, False, info

