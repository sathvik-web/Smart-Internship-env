---

title: "Smart Internship Env"
emoji: "🤖"
colorFrom: "yellow"
colorTo: "red"
sdk: "docker"
app_file: "app.py"
pinned: false
-------------

# SmartInternshipEnv

## Live Demo

🔗 https://sathvik1890-smart-internship-env-final.hf.space

---

## Overview

SmartInternshipEnv is an OpenEnv-compatible environment designed to evaluate how an AI agent makes internship-related decisions based on skill alignment.

It simulates real-world decision-making scenarios where an agent must:

* decide whether to apply for an internship
* assign a relevance score
* rank multiple internship options

The environment focuses on **agent evaluation**, not recommendation, using deterministic grading and structured reward signals.

---

## Tech Stack

* Python
* FastAPI
* Pydantic
* Docker
* Hugging Face Spaces
* OpenEnv

---

## Environment Design

### Core API

* `reset()` → returns initial observation
* `step(action)` → returns `(observation, reward, done, info)`
* `state()` → returns current environment state

---

### Models

* **Observation** → internship details + student skills
* **Action** → decision, relevance score, ranking, reasoning
* **Reward** → normalized evaluation score

---

## Tasks

The environment includes **exactly 3 tasks**, each with a **deterministic grader**:

* **Easy** → classification (apply / ignore)
* **Medium** → relevance scoring (0.0 to 1.0)
* **Hard** → ranking multiple internship options

Each task increases in complexity and evaluates different aspects of agent decision-making.

---

## Reward Design

* Reward is normalized in **[0.0, 1.0]**
* Provides **partial credit (not binary)**
* Evaluates:

  * decision correctness
  * relevance score accuracy
  * ranking quality (hard task)
  * reasoning consistency

---

## API Endpoints

| Endpoint  | Method   | Description                 |
| --------- | -------- | --------------------------- |
| `/reset`  | GET/POST | Start a new task            |
| `/step`   | POST     | Execute agent action        |
| `/tasks`  | GET      | List all tasks with graders |
| `/state`  | GET/POST | Get environment state       |
| `/health` | GET      | Health check                |

---

## API Verification

The deployed environment exposes:

* `/tasks` → returns **3 tasks with graders**
* `/reset` → returns valid observation
* `/step` → returns reward, done, info

All endpoints are accessible via the deployed Hugging Face Space.

---

## Inference

The environment includes a baseline inference script.

### Run:

```bash
python inference.py
```

### Log Format:

```
[START] task=... env=... model=...
[STEP] step=... action=... reward=... done=... error=...
[END] success=... steps=... score=... rewards=...
```

* Structured logs
* Deterministic execution
* Scores normalized between 0 and 1

---

## Run Locally

```bash
pip install -r requirements.txt
python inference.py
```

---

## Docker

```bash
docker build -t smart-internship-env .
docker run -p 7860:7860 smart-internship-env
```

---

## Deployment

The environment is deployed using Docker on Hugging Face Spaces.

🔗 https://sathvik1890-smart-internship-env-final.hf.space

---

## Validation

```bash
openenv validate
```

---

## Features

* Real-world decision-making simulation
* Deterministic grading system
* Multi-step environment interaction
* Reward shaping with partial signals
* OpenEnv compliant
* Fully deployed and reproducible

---

## Deployment Status

* Hugging Face Space: ✅ Running
* Docker: ✅ Working
* OpenEnv Validation: ✅ Passed
* Tasks with Graders: ✅ Verified (3/3)

---

## Outcome

This project demonstrates a robust AI evaluation environment where agents are tested on realistic decision-making tasks involving classification, scoring, and ranking. It provides a structured framework for assessing agent performance using deterministic rewards and reproducible execution.
