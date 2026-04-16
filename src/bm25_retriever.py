from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHROMA_DIR  = config["chroma_dir"]
EMBED_MODEL = config["embed_model"]
TOP_K       = config["top_k"]

def load_all_chunks():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    all_docs = vectorstore.get()
    return all_docs

def build_bm25_index(all_docs):
    documents = all_docs["documents"]
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, documents  

def bm25_search(query, bm25, documents, all_docs, k=TOP_K):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    results = []
    for i in top_k_indices:
        results.append({
            "content": documents[i],
            "metadata": all_docs["metadatas"][i],
            "score": scores[i]
        })
    return results

# test the BM25 retriever with a sample query
# if __name__ == "__main__":

#     all_docs = load_all_chunks()
#     bm25, documents = build_bm25_index(all_docs)
#     query = "How does LoRA work?"
#     print(f"\nSearching for: '{query}'")
#     results = bm25_search(query, bm25, documents, all_docs)

#     print(f"\nTop {TOP_K} results:")
#     for i, result in enumerate(results):
#         print(f"\n[{i+1}] {result['metadata']} — score: {result['score']:.4f}")
#         print(f"    {result['content'][:200]}...")