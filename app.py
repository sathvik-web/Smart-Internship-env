from fastapi import FastAPI
from env.environment import InternshipEnv

# THIS LINE IS CRITICAL
app = FastAPI()

env = InternshipEnv()

@app.get("/")
def root():
    return {"message": "SmartInternshipEnv is running"}

@app.get("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.get("/state")
def state():
    return env.state()