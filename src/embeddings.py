from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class LocalEmbeddings(Embeddings):
    def __init__(self, name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()
