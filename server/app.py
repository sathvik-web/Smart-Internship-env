from fastapi import FastAPI
from env.environment import InternshipEnv
from env.models import Action
from env.tasks import load_tasks

app = FastAPI()

env = InternshipEnv()


@app.get("/")
def root():
    return {"message": "SmartInternshipEnv is running"}


@app.get("/tasks")
@app.post("/tasks")
def tasks():
    task_list = load_tasks()
    return [
        {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "grader": callable(getattr(task, "grader", None)),
        }
        for task in task_list
    ]


@app.post("/reset")
@app.get("/reset")
def reset():
    try:
        observation = env.reset()
        return {"observation": observation.model_dump(exclude={"grader"})}
    except Exception as exc:
        return {"observation": None, "error": str(exc)}


@app.post("/step")
def step(action: dict):
    try:
        parsed_action = Action.model_validate(action)
        observation, reward, done, info = env.step(parsed_action)
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

def main():
    return InternshipEnv()

if __name__ == "__main__":
    main()