from fastapi import FastAPI
from env.environment import InternshipEnv
from env.models import Action

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


@app.get("/state")
@app.post("/state")
def state():
    try:
        return env.state()
    except Exception as exc:
        return {"error": str(exc)}


def main():
    return InternshipEnv()


if __name__ == "__main__":
    main()