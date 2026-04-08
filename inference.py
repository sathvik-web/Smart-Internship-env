from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from env.environment import InternshipEnv
from env.models import Action, Observation

MAX_STEPS = 16
ENV_NAME = "SmartInternshipEnv"


def _extract_json_object(raw_text: str) -> dict[str, Any]:
	raw_text = raw_text.strip()
	if raw_text.startswith("```"):
		raw_text = raw_text.strip("`")
		if raw_text.lower().startswith("json"):
			raw_text = raw_text[4:].strip()

	if raw_text.startswith("{") and raw_text.endswith("}"):
		return json.loads(raw_text)

	first_brace = raw_text.find("{")
	last_brace = raw_text.rfind("}")
	if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
		raise ValueError("No JSON object found in model output")

	return json.loads(raw_text[first_brace : last_brace + 1])


def _heuristic_action(observation: Observation) -> Action:
	required = {skill.lower() for skill in observation.required_skills}
	student = {skill.lower() for skill in observation.student_skills}
	overlap = len(required.intersection(student))
	coverage = overlap / len(required) if required else 1.0

	decision = "apply" if coverage >= 0.45 else "ignore"
	ranking = []
	top_coverage = coverage
	if observation.internship_options:
		scored = []
		for option in observation.internship_options:
			option_required = {skill.lower() for skill in option.required_skills}
			option_coverage = (
				len(option_required.intersection(student)) / len(option_required) if option_required else 1.0
			)
			scored.append((option.internship_title, option_coverage))
		scored.sort(key=lambda item: (-item[1], item[0]))
		ranking = [title for title, _ in scored]
		top_coverage = scored[0][1]
		decision = "apply" if top_coverage >= 0.50 else "ignore"

	relevance_score = round(max(0.0, min(1.0, top_coverage if ranking else coverage)), 3)

	return Action(
		decision=decision,
		relevance_score=relevance_score,
		ranking=ranking,
		reasoning=(
			"Decision uses deterministic skill overlap, domain fit, and ranking of internships by"
			" matched required skills."
		),
	)


def _build_prompt(observation: Observation) -> str:
	schema_hint = {
		"decision": "apply | ignore",
		"relevance_score": "float between 0 and 1",
		"ranking": ["internship title", "..."],
		"reasoning": "brief explanation",
	}
	return (
		"You are an internship recommendation assistant. "
		"Return only one JSON object with this schema: "
		f"{json.dumps(schema_hint)}\n"
		f"Task: {observation.model_dump_json()}"
	)


def _model_action(client: OpenAI, model_name: str, observation: Observation) -> Action:
	response = client.chat.completions.create(
		model=model_name,
		temperature=0,
		messages=[
			{
				"role": "system",
				"content": (
					"You output valid JSON only. Keep scores calibrated to skill-fit realism "
					"and include ranking for hard tasks when options are present."
				),
			},
			{"role": "user", "content": _build_prompt(observation)},
		],
	)

	content = response.choices[0].message.content or ""
	payload = _extract_json_object(content)
	return Action.model_validate(payload)


def main() -> int:
	load_dotenv()

	api_base_url = os.getenv("API_BASE_URL", "").strip()
	model_name = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()
	hf_token = os.getenv("HF_TOKEN")

	env = InternshipEnv()
	observation = env.reset()

	client = OpenAI(base_url=api_base_url or None, api_key=hf_token or "DUMMY")

	rewards: list[float] = []
	success = True
	step_idx = 0

	print(f"[START] task=all env={ENV_NAME} model={model_name}")

	done = False
	while not done and step_idx < MAX_STEPS:
		step_idx += 1
		error_msg = "null"

		try:
			if hf_token:
				action = _model_action(client, model_name, observation)
			else:
				action = _heuristic_action(observation)
		except (ValueError, ValidationError, KeyError, IndexError) as exc:
			action = _heuristic_action(observation)
		except Exception as exc:  # noqa: BLE001
			action = _heuristic_action(observation)

		try:
			observation, reward, done, _info = env.step(action)
			rewards.append(reward)
			action_log = json.dumps(action.model_dump(), separators=(",", ":"), sort_keys=True)
			print(
				f"[STEP] step={step_idx} action={action_log} "
				f"reward={reward:.2f} done={str(done).lower()} error={error_msg}"
			)
		except Exception as exc:  # noqa: BLE001
			success = False
			print(
				f"[STEP] step={step_idx} action=none reward=0.00 done=true "
				f"error=env_error:{type(exc).__name__}"
			)
			break

	if not done:
		success = False

	final_score = sum(rewards) / len(rewards) if rewards else 0.0
	rewards_log = ",".join(f"{item:.2f}" for item in rewards)
	print(
		f"[END] success={str(success).lower()} steps={step_idx} "
		f"score={final_score:.2f} rewards={rewards_log}"
	)

	return 0 if success else 1


if __name__ == "__main__":
	raise SystemExit(main())

