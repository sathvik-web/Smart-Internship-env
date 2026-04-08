---
title: "Smart Internship Env"
colorFrom: "yellow"
colorTo: "red"
sdk: "docker"
sdk_version: "latest"
app_file: "app.py"
pinned: false
---
# SmartInternshipEnv

## Live Demo

https://huggingface.co/spaces/sathvik1890/smart-internship-env

## Overview

SmartInternshipEnv is a real-world OpenEnv environment that simulates internship recommendation based on student skills.
It evaluates an AI agent’s ability to make decisions such as applying, assigning relevance scores, and ranking multiple internship opportunities.

This environment is designed for agent evaluation, decision-making, and reward-based learning, similar to real-world job recommendation systems.

---

## Tech Stack

* Python
* OpenEnv
* FastAPI
* Docker
* Hugging Face Spaces

---

## Environment Design

### Core API

* `reset()` → returns initial observation
* `step(action)` → returns observation, reward, done, info
* `state()` → returns current state

### Models

* `Observation` → internship + student details
* `Action` → decision, score, ranking, reasoning
* `Reward` → normalized score

---

## Tasks

The environment includes three deterministic tasks:

* Easy → classification (apply / ignore)
* Medium → relevance scoring (0.0 to 1.0)
* Hard → ranking multiple internships

Tasks increase in complexity and simulate real-world decision-making.

---

## Reward Design

* Reward is normalized in [0, 1]
* Provides partial credit
* Evaluates:

  * decision correctness
  * score calibration
  * ranking accuracy (hard task)
  * reasoning quality

---

## Inference

The environment includes a baseline inference script using the OpenAI client.

### Log Format

```
[START] task=... env=... model=...
[STEP] step=... action=... reward=... done=... error=...
[END] success=... steps=... score=... rewards=...
```

* Rewards formatted to 2 decimals
* Scores normalized between 0 and 1
* Deterministic and reproducible

---

## Run Locally

```bash
pip install -r requirements.txt
python inference.py
```

---

## Validation

```bash
openenv validate
```

---

## Docker

```bash
docker build -t smart-internship-env .
docker run smart-internship-env
```

---

## Deployment

The environment is deployed on Hugging Face Spaces using Docker and FastAPI.

---

## Features

* Real-world task simulation
* Deterministic grading
* Multi-step decision evaluation
* Reward shaping with partial signals
* OpenEnv compliant
* Fully deployed and reproducible

---

## Outcome

This project demonstrates an AI evaluation system where agents are tested on realistic decision-making scenarios involving classification, scoring, and ranking.
