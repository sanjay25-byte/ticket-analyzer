import pickle
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from src.embeddings import LocalEmbeddings

_emb = LocalEmbeddings()
_ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def tok(t):
    return re.findall(r"\w+", t.lower())


def rrf(lists, k=60):
    """Reciprocal Rank Fusion over several ranked lists of Documents."""
    scores, docs = {}, {}
    for lst in lists:
        for rank, d in enumerate(lst):
            key = d.page_content
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            docs[key] = d
    return [docs[x] for x in sorted(scores, key=scores.get, reverse=True)]


class Retriever:
    def __init__(self, strategy):
        self.db = Chroma(collection_name=strategy, embedding_function=_emb,
                         persist_directory=f"chroma_db/{strategy}")
        self.chunks = pickle.load(open(f"chroma_db/{strategy}.pkl", "rb"))
        self.bm25 = BM25Okapi([tok(c.page_content) for c in self.chunks])

    def _vector(self, q, n, cat):
        return self.db.similarity_search(q, k=n, filter={"category": cat} if cat else None)

    def _bm25(self, q, n, cat):
        s = self.bm25.get_scores(tok(q))
        order = sorted(range(len(s)), key=lambda i: s[i], reverse=True)
        return [self.chunks[i] for i in order
                if not cat or self.chunks[i].metadata["category"] == cat][:n]

    def search(self, q, mode="hybrid_rerank", k=4, category=None, n=20):
        if mode == "vector":
            docs = self._vector(q, n, category)
        elif mode == "bm25":
            docs = self._bm25(q, n, category)
        else:
            docs = rrf([self._vector(q, n, category), self._bm25(q, n, category)])[:n]
            if mode == "hybrid_rerank" and docs:
                s = _ce.predict([(q, d.page_content) for d in docs])
                docs = [d for _, d in sorted(zip(s, docs), key=lambda x: x[0], reverse=True)]
        return self._finalize(docs, k)

    def _finalize(self, docs, k):
        out, seen = [], set()
        for d in docs:
            tid = d.metadata["ticket_id"]
            if tid in seen:
                continue
            seen.add(tid)
            meta = {m: v for m, v in d.metadata.items() if m != "parent_text"}
            out.append(Document(page_content=d.metadata.get("parent_text", d.page_content),
                                metadata=meta))
            if len(out) == k:
                break
        return out


@lru_cache(maxsize=None)
def get_retriever(strategy):
    return Retriever(strategy)
