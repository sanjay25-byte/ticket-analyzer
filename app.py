import streamlit as st
from src.agent import ask

st.title("AI Ticket Analyzer")
q = st.text_input("Ask about tickets, fixes or ticket stats")

if st.button("Ask") and q:
    with st.spinner("Thinking..."):
        r = ask(q)
    if r["route"] == "data":
        st.code(r["answer"])
    else:
        st.write(r["answer"])
    a, b, c = st.columns(3)
    a.metric("Route", r["route"])
    b.metric("Confidence", "-" if r["confidence"] is None else f'{r["confidence"]:.2f}')
    c.metric("Latency", f'{r["latency_s"]}s')
    st.caption("Citations: " + (", ".join(r["citations"]) or "none"))
    if r["escalated"]:
        st.warning("Escalated to a human agent")
    with st.expander("Token usage"):
        st.json(r["usage"])