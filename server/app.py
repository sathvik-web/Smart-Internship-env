from fastapi import FastAPI
from env.environment import InternshipEnv

app = FastAPI()

env = InternshipEnv()


@app.get("/")
def root():
    return {"message": "SmartInternshipEnv is running"}


@app.post("/reset")
@app.get("/reset")
def reset():
    observation = env.reset()
    return {"observation": observation}


@app.post("/step")
def step(action: dict):
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info
    }

def main():
    return InternshipEnv()

if __name__ == "__main__":
    main()