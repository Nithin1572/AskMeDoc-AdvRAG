from sentence_transformers import CrossEncoder
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

TOP_K = config["top_k"]

def load_reranker():
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return model

def rerank(query, chunks, model):
    pairs = [(query, doc.page_content) for doc in chunks]
    scores = model.predict(pairs)
    
    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    reranked_chunks = [doc for doc, score in ranked]
    return reranked_chunks