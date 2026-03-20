from fastapi import FastAPI
import os

app = FastAPI(title="Signal Backend")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Signal Backend is running!"}
