import os
import pickle
import shutil
import sys

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from src.embeddings import LocalEmbeddings

STRATEGIES = ["fixed", "recursive", "semantic", "parent_child"]


def load_tickets():
    df = pd.read_csv("data/tickets.csv")
    return [Document(
        page_content=f"Title: {r.title}\nDescription: {r.description}\nResolution: {r.resolution}",
        metadata=dict(ticket_id=r.ticket_id, title=r.title, category=r.category, priority=r.priority))
        for r in df.itertuples()]


def load_kb():
    text = open("data/knowledge_base.txt", encoding="utf-8").read()
    docs = []
    for i, part in enumerate(text.split("## ")[1:]):
        head, _, body = part.partition("\n")
        title, cat = [x.strip() for x in head.split("|")]
        docs.append(Document(
            page_content=f"{title}\n{body.strip()}",
            metadata=dict(ticket_id=f"KB{i + 1}", title=title, category=cat, priority="NA")))
    return docs


def chunk(docs, strategy):
    if strategy == "fixed":
        return CharacterTextSplitter(separator="", chunk_size=150, chunk_overlap=0).split_documents(docs)
    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30).split_documents(docs)
    if strategy == "semantic":
        return SemanticChunker(LocalEmbeddings()).split_documents(docs)
    # parent_child: index small chunks, return the full parent text at retrieval
    sp, out = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20), []
    for d in docs:
        for c in sp.split_documents([d]):
            c.metadata["parent_text"] = d.page_content
            out.append(c)
    return out


def build(strategy):
    chunks = chunk(load_tickets() + load_kb(), strategy)
    os.makedirs("chroma_db", exist_ok=True)
    path = f"chroma_db/{strategy}"
    shutil.rmtree(path, ignore_errors=True)
    Chroma.from_documents(chunks, LocalEmbeddings(), collection_name=strategy, persist_directory=path)
    pickle.dump(chunks, open(f"chroma_db/{strategy}.pkl", "wb"))
    print(strategy, len(chunks), "chunks")


if __name__ == "__main__":
    for s in (sys.argv[1:] or STRATEGIES):
        build(s)
