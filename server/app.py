from fastapi import FastAPI
from fastapi.responses import JSONResponse

from env.environment import InternshipEnv
from env.models import Action
from env.tasks import load_tasks

print("SERVER.APP LOADED")

app = FastAPI(docs_url=None, redoc_url=None)
env = InternshipEnv()


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(content={"message": "SmartInternshipEnv is running"})


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


@app.post("/reset")
def reset(task: str | None = None) -> JSONResponse:
    print("RESET CALLED")
    observation = env.reset(task=task)
    return JSONResponse(content={"observation": observation.model_dump(exclude={"grader"})})


@app.post("/step")
def step(action: Action) -> JSONResponse:
    observation, reward, done, info = env.step(action)
    return JSONResponse(
        content={
            "observation": None if observation is None else observation.model_dump(exclude={"grader"}),
            "reward": float(reward),
            "done": bool(done),
            "info": dict(info),
        }
    )


@app.get("/tasks")
def tasks() -> JSONResponse:
    print("TASKS CALLED")
    tasks_list = load_tasks()
    tasks_payload = [
        {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "grader": True,
        }
        for task in tasks_list
    ]
    return JSONResponse(content=tasks_payload)