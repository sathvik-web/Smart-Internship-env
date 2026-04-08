from fastapi import FastAPI
from env.environment import InternshipEnv
from env.models import Action

app = FastAPI()

env = InternshipEnv()


@app.get("/")
def root():
    return {"message": "SmartInternshipEnv is running"}


@app.post("/reset")
@app.get("/reset")
def reset():
    observation = env.reset()
    return {"observation": observation.model_dump(exclude={"grader"})}


@app.post("/step")
def step(action: dict):
    parsed_action = Action.model_validate(action)
    observation, reward, done, info = env.step(parsed_action)
    return {
        "observation": observation.model_dump(exclude={"grader"}),
        "reward": reward,
        "done": done,
        "info": info
    }

def main():
    return InternshipEnv()

if __name__ == "__main__":
    main()