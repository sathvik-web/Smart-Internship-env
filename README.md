# SmartInternshipEnv

SmartInternshipEnv is a production-oriented OpenEnv environment for internship recommendation and decision-making.

It evaluates whether an agent can make realistic internship decisions from student skill profiles using deterministic grading and normalized rewards.

## Why This Environment

Internship matching in practice is not a binary filter. Strong recommendation systems need to:

- classify obvious fits and non-fits
- estimate nuanced relevance under partial overlap
- rank multiple options by best overall fit
- justify decisions with concrete evidence

SmartInternshipEnv is designed to test exactly these abilities.

## Environment Interface

`InternshipEnv` provides an OpenEnv-style API:

- `reset() -> Observation`
- `step(action: Action) -> (Observation, reward, done, info)`
- `state() -> dict`

The environment is deterministic and runs sequentially through fixed tasks.

## Observation Space

`Observation` includes:

- `task_id`
- `difficulty` (`easy`, `medium`, `hard`)
- `objective`
- `internship_title`
- `description`
- `required_skills`
- `student_skills`
- `internship_options` (used by hard ranking)

## Action Space

`Action` includes:

- `decision`: `apply` or `ignore`
- `relevance_score`: float in `[0.0, 1.0]`
- `ranking`: ordered internship titles (important for hard task)
- `reasoning`: short textual rationale

## Task Design

The environment contains three deterministic tasks:

1. Easy: Classification
- One backend internship
- Objective: output `apply` or `ignore`

2. Medium: Nuanced Scoring
- MLOps-focused internship with partial skill overlap
- Objective: output calibrated relevance score and decision

3. Hard: Multi-Option Ranking
- Four realistic internships across ML, Backend, Data, and Frontend
- Objective: rank all options, decide on top option, and provide top-option relevance

Difficulty increases from straightforward matching to tradeoff-heavy ranking.

## Reward Design

Reward is deterministic and normalized to `[0, 1]`.

Core reward factors:

- decision correctness
- score closeness (`1 - |predicted - true_score|`)
- reasoning keyword coverage
- ranking quality for hard task (pairwise ordering)
- progress bonus

Penalties include:

- wrong decision (stronger for overconfident outputs)
- extreme score mismatch
- poor reasoning quality
- missing ranking for hard task

This provides smooth partial credit while discouraging brittle outputs.

## Inference Runtime

`inference.py`:

- uses OpenAI client
- reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- runs tasks sequentially with `MAX_STEPS`
- falls back to deterministic heuristic actions when model calls fail
- parses JSON robustly from model output

### Required Log Format

- `[START] task=... env=... model=...`
- `[STEP] step=... action=... reward=... done=... error=...`
- `[END] success=... steps=... score=... rewards=...`

Step rewards are printed to **2 decimal places**.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment variables (Windows):

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
set HF_TOKEN=your_token_here
```

For Linux/macOS, use `export`.

## Run

```bash
python inference.py
```

## Docker

```bash
docker build -t smart-internship-env .
docker run --rm \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_token_here \
  smart-internship-env
```

## Expected Output

```text
[START] task=all env=SmartInternshipEnv model=gpt-4o-mini
[STEP] step=1 action={...} reward=0.78 done=false error=none
[STEP] step=2 action={...} reward=0.63 done=false error=none
[STEP] step=3 action={...} reward=0.84 done=true error=none
[END] success=true steps=3 score=0.75 rewards=[0.78,0.63,0.84]
```
