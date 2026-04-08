---
title: Smart Internship Env
emoji: 🏃
colorFrom: yellow
colorTo: red
sdk: docker
sdk_version: "latest"
app_file: app.py
pinned: false
-------------

# SmartInternshipEnv

SmartInternshipEnv is a deterministic OpenEnv environment for internship recommendation and ranking based on student skills.

It is designed to be submission-ready for OpenEnv evaluation and deployment-ready for Hugging Face Spaces (Docker).

## Environment Summary

Core interface (`env/environment.py`):

- `reset() -> Observation`
- `step(action: Action) -> (Observation, reward, done, info)`
- `state() -> dict`

Typed models (`env/models.py`):

- `Observation`
- `Action`
- `Reward`

## Task Design

The environment includes three deterministic tasks:

1. Easy: classification (`apply` vs `ignore`)
2. Medium: nuanced relevance scoring in `[0.0, 1.0]`
3. Hard: ranking multiple internships across different domains

The hard task is graded with pairwise ranking correctness.

## Reward Design

Reward is always normalized to `[0, 1]` and includes:

- partial credit for close relevance scores
- decision correctness signal
- reasoning quality via deterministic keyword matching
- ranking quality for the hard task
- progress bonus across episode steps
- penalties for wrong decisions and poor calibration/reasoning

## Inference Logging Format

`inference.py` prints strict logs:

- `[START] task=... env=... model=...`
- `[STEP] step=... action=... reward=... done=... error=...`
- `[END] success=... steps=... score=... rewards=...`

Formatting guarantees:

- `reward` uses exactly 2 decimals
- `score` uses exactly 2 decimals
- `done` and `success` are lowercase booleans (`true`/`false`)
- `error=null` when no runtime error
- `rewards` is comma-separated with no brackets

Example:

```text
[START] task=all env=SmartInternshipEnv model=gpt-4o-mini
[STEP] step=1 action={...} reward=0.78 done=false error=null
[STEP] step=2 action={...} reward=0.73 done=false error=null
[STEP] step=3 action={...} reward=0.87 done=true error=null
[END] success=true steps=3 score=0.79 rewards=0.78,0.73,0.87
```

## Packaging and Validation

The project includes OpenEnv packaging requirements:

- `pyproject.toml` includes `openenv-core>=0.2.0`
- script entrypoint:
  - `server = "server.app:main"`
- `server/app.py` defines callable `main()`
- `env/__init__.py` defines `env` as a proper package

Lockfile support:

```bash
uv lock
```

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables (Windows):

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
set HF_TOKEN=your_token_here
```

Run inference:

```bash
python inference.py
```

Validate OpenEnv package:

```bash
openenv validate
```

## Docker / Hugging Face Spaces

Build image:

```bash
docker build -t smart-internship-env .
```

Run container:

```bash
docker run --rm \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_token_here \
  smart-internship-env
```
