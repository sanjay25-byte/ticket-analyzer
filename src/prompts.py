PROMPT_VERSION = "v1"

ROUTER = """Classify this IT support question.
- ticket_kb: how to fix an issue, troubleshooting, similar past tickets, policies
- data: counting, aggregating or listing tickets (e.g. how many P1 tickets, most common category)
- human: needs a human action, or is unrelated or unsafe
Also give category if clearly one of VPN, Password, Email, Printer, Software, Hardware, Access; otherwise null.
Question: {q}"""

REWRITE = """Rewrite as a short keyword-rich search query for an IT ticket knowledge base.
Keep error codes verbatim. Return only the query.
Question: {q}
Previous query: {prev}"""

GRADE = """Question: {q}

Context:
{ctx}

Is the context relevant and sufficient to answer the question? Be strict."""

ANSWER = """Answer using ONLY the context. List the IDs you used (e.g. INC1001, KB1) in citations.
confidence is 0-1 for how well the context supports the answer. If the context lacks the answer, say so and use 0.

Context:
{ctx}

Question: {q}"""

SQL = """Table tickets(ticket_id TEXT, category TEXT, priority TEXT, title TEXT, description TEXT, resolution TEXT).
Categories: VPN, Password, Email, Printer, Software, Hardware, Access. Priorities: P1-P4.
Write ONE SQLite SELECT query that answers the question. Return only the SQL.
Question: {q}"""
