from src.retrieval import load_vectorstore, retrieve_chunks, build_prompt
from src.bm25_retriever import load_all_chunks, build_bm25_index, bm25_search
from src.reranker import load_reranker, rerank
from sentence_transformers import CrossEncoder
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

LLM_MODEL = config["llm_model"]
TOP_K = config["top_k"]

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores = {}
    documents = {}

    for rank, doc in enumerate(vector_results):
        doc_id = doc.page_content
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        documents[doc_id] = doc

    for rank, result in enumerate(bm25_results):
        doc_id = result["content"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        if doc_id not in documents:
            documents[doc_id] = Document(
                page_content=result["content"],
                metadata=result["metadata"]
            )

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [documents[doc_id] for doc_id in sorted_ids[:TOP_K]]

def hybrid_search(query):
    vectorstore = load_vectorstore()
    vector_results = retrieve_chunks(query, vectorstore)

    all_docs = load_all_chunks()
    bm25, documents = build_bm25_index(all_docs)
    bm25_results = bm25_search(query, bm25, documents, all_docs)

    fused_results = reciprocal_rank_fusion(vector_results, bm25_results)

    return fused_results

def ask(query, model):
    chunks = hybrid_search(query)
    reranked_chunks = rerank(query, chunks, model)
    prompt = build_prompt(query, reranked_chunks)
    llm = OllamaLLM(model=LLM_MODEL)
    answer = llm.invoke(prompt)
    return answer, reranked_chunks