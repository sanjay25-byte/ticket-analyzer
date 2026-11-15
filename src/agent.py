import csv
import os
import re
import sqlite3
import time
import uuid
from typing import Literal, Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langchain_core.callbacks import UsageMetadataCallbackHandler

from src.llm import llm
from src import prompts as P
from src.retriever import get_retriever

STRATEGY = os.getenv("CHUNK_STRATEGY", "parent_child")
CATS = {"VPN", "Password", "Email", "Printer", "Software", "Hardware", "Access"}


class Route(BaseModel):
    route: Literal["ticket_kb", "data", "human"]
    category: Optional[str] = None


class Grade(BaseModel):
    relevant: bool
    reason: str


class Answer(BaseModel):
    answer: str
    citations: list[str]
    confidence: float = Field(ge=0, le=1)


class State(TypedDict, total=False):
    question: str
    route: str
    category: Optional[str]
    query: str
    docs: list
    attempts: int
    relevant: bool
    answer: str
    citations: list
    confidence: float
    escalated: bool
    reason: str


def fmt(docs):
    return "\n\n".join(f"[{d.metadata['ticket_id']}] {d.page_content}" for d in docs)


def route(s):
    r = llm.with_structured_output(Route).invoke(P.ROUTER.format(q=s["question"]))
    return {"route": r.route,
            "category": r.category if r.category in CATS else None,
            "attempts": 0,
            "query": s["question"],
            "reason": "routed to human" if r.route == "human" else ""}


def rewrite(s):
    q = llm.invoke(P.REWRITE.format(q=s["question"], prev=s["query"])).content.strip()
    return {"query": q}


def retrieve(s):
    cat = s.get("category") if s["attempts"] == 0 else None  # drop the filter on retry
    docs = get_retriever(STRATEGY).search(s["query"], mode="hybrid_rerank", k=4, category=cat)
    return {"docs": docs, "attempts": s["attempts"] + 1}


def grade(s):
    g = llm.with_structured_output(Grade).invoke(
        P.GRADE.format(q=s["question"], ctx=fmt(s["docs"])))
    return {"relevant": g.relevant}


def after_grade(s):
    if s["relevant"]:
        return "answer"
    return "rewrite" if s["attempts"] < 2 else "escalate"


def answer(s):
    ids = {d.metadata["ticket_id"] for d in s["docs"]}
    out = llm.with_structured_output(Answer).invoke(
        P.ANSWER.format(q=s["question"], ctx=fmt(s["docs"])))
    cites = [c.strip("[] ") for c in out.citations if c.strip("[] ") in ids]  # citation guardrail
    return {"answer": out.answer, "citations": cites,
            "confidence": out.confidence if cites else 0.0,
            "reason": "low confidence or missing citations"}


def sql(s):
    text = llm.invoke(P.SQL.format(q=s["question"])).content
    q = re.sub(r"```(sql)?", "", text).strip().rstrip(";")
    if not re.fullmatch(r"(?is)select\b[^;]*", q):
        return {"answer": "Could not build a safe query.", "citations": []}
    con = sqlite3.connect(":memory:")
    pd.read_csv("data/tickets.csv").to_sql("tickets", con, index=False)
    rows = pd.read_sql(q, con).head(20)
    return {"answer": f"SQL: {q}\n\n{rows.to_string(index=False)}", "citations": []}


def escalate(s):
    tid = f"ESC{uuid.uuid4().hex[:6].upper()}"
    new = not os.path.exists("data/escalations.csv")
    with open("data/escalations.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "question", "reason"])
        w.writerow([tid, s["question"], s.get("reason") or "no relevant context"])
    msg = ("This needs a human, so I created ticket " if s.get("route") == "human"
           else "I couldn't answer this reliably, so I created ticket ")
    return {"answer": f"{msg}{tid} for a human agent.", "escalated": True, "citations": []}

g = StateGraph(State)
for name, fn in [("n_route", route), ("n_rewrite", rewrite), ("n_retrieve", retrieve),
                 ("n_grade", grade), ("n_answer", answer), ("n_sql", sql),
                 ("n_escalate", escalate)]:
    g.add_node(name, fn)

g.add_edge(START, "n_route")
g.add_conditional_edges("n_route", lambda s: s["route"],
                        {"ticket_kb": "n_rewrite", "data": "n_sql", "human": "n_escalate"})
g.add_edge("n_rewrite", "n_retrieve")
g.add_edge("n_retrieve", "n_grade")
g.add_conditional_edges("n_grade", after_grade,
                        {"answer": "n_answer", "rewrite": "n_rewrite", "escalate": "n_escalate"})
g.add_conditional_edges("n_answer", lambda s: "end" if s["confidence"] >= 0.5 else "escalate",
                        {"end": END, "escalate": "n_escalate"})
g.add_edge("n_sql", END)
g.add_edge("n_escalate", END)
graph = g.compile()


def ask(question):
    cb, t = UsageMetadataCallbackHandler(), time.time()
    s = graph.invoke({"question": question}, config={"callbacks": [cb]})
    return {"answer": s["answer"], "route": s.get("route"),
            "citations": s.get("citations", []), "confidence": s.get("confidence"),
            "escalated": s.get("escalated", False),
            "latency_s": round(time.time() - t, 2), "usage": cb.usage_metadata,
            "prompt_version": P.PROMPT_VERSION}
