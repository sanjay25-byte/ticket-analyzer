from fastapi import FastAPI
from pydantic import BaseModel
from src.agent import ask

app = FastAPI(title="AI Ticket Analyzer")


class Q(BaseModel):
    question: str


@app.post("/ask")
def ask_endpoint(q: Q):
    return ask(q.question)
