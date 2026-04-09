from fastapi import FastAPI
from env.environment import InternshipEnv
from env.models import Action
from env.tasks import load_tasks

print(" SERVER.APP LOADED WITH TASKS ROUTE")

app = FastAPI()

env = InternshipEnv()


@app.get("/")
def root():
    return {"message": "SmartInternshipEnv is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
@app.get("/reset")
def reset(task: str = None):
    try:
        observation = env.reset(task=task)
        return {"observation": observation.model_dump(exclude={"grader"})}
    except Exception as exc:
        return {"observation": None, "error": str(exc)}


@app.post("/step")
def step(action: dict):
    try:
        parsed = Action.model_validate(action)
        observation, reward, done, info = env.step(parsed)
        return {
            "observation": None if observation is None else observation.model_dump(exclude={"grader"}),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as exc:
        return {
            "observation": None,
            "reward": 0.0,
            "done": True,
            "info": {"error": str(exc)},
        }


@app.get("/state")
@app.post("/state")
def state():
    try:
        return env.state()
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/tasks")
@app.post("/tasks")
def tasks():
    tasks = load_tasks()
    print(" TASKS LOADED:", len(tasks))
    print(" GRADER FLAGS:", [callable(getattr(t, "grader", None)) for t in tasks])

    return [
        {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "grader": callable(getattr(task, "grader", None)),
        }
        for task in tasks
    ]


print(" TASKS ROUTE REGISTERED")


# DEBUG ROUTES (CRITICAL)
@app.get("/debug-routes")
def debug_routes():
    return [route.path for route in app.routes]


def main():
    return InternshipEnv()


if __name__ == "__main__":
    main()