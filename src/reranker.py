from sentence_transformers import CrossEncoder
from hybrid_retriever import hybrid_search
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

TOP_K = config["top_k"]

def load_reranker():
    print("Loading re-ranker model...")
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Re-ranker model loaded!")
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

# test the re-ranker with a sample query and retrieved chunks
# if __name__ == "__main__":
#     query = "How does LoRA work?"
#     print(f"Query: {query}\n")

#     print("Running hybrid search...")
#     chunks = hybrid_search(query)

#     print("\nBefore re-ranking:")
#     for i, doc in enumerate(chunks):
#         source = doc.metadata.get("filename", "unknown")
#         page = doc.metadata.get("page", "?")
#         print(f"  [{i+1}] {source} — page {page}")

#     model = load_reranker()
#     reranked = rerank(query, chunks, model)

#     print("\nAfter re-ranking:")
#     for i, doc in enumerate(reranked):
#         source = doc.metadata.get("filename", "unknown")
#         page = doc.metadata.get("page", "?")
#         print(f"  [{i+1}] {source} — page {page}")
#         print(f"      {doc.page_content[:200]}...")