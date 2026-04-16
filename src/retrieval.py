from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHROMA_DIR  = config["chroma_dir"]
EMBED_MODEL = config["embed_model"]
LLM_MODEL   = config["llm_model"]
TOP_K       = config["top_k"]

def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def retrieve_chunks(query, vectorstore):
    results = vectorstore.similarity_search(query, k=TOP_K)
    return results

def build_prompt(query, chunks):
    context = ""
    for i, doc in enumerate(chunks):
        source = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "?")
        context += f"\n[{i+1}] (Source: {source}, Page: {page})\n{doc.page_content}\n"

    prompt = f"""You are a helpful research assistant. Answer the question based ONLY on the context provided below. If the answer is 
    not found in the context, say "I don't know based on the provided documents." Do NOT use any outside knowledge. You MUST only 
    cite source numbers that appear in the context below, which are [1] to [{len(chunks)}]. Never invent or use source numbers 
    outside this range. Context: {context} Question: {query} Answer:"""
    return prompt

def ask(query):
    vectorstore = load_vectorstore()
    chunks = retrieve_chunks(query, vectorstore)
    prompt = build_prompt(query, chunks)
    
    llm = OllamaLLM(model=LLM_MODEL)
    answer = llm.invoke(prompt)
    
    return answer, chunks

# test the retrieval and prompting with a sample query
# if __name__ == "__main__":
#     # query = "What is the transformer architecture?"
#     query = "What is 1 + 1= ?"

#     print(f"\nQuestion: {query}\n")
#     answer, chunks = ask(query)

#     print(f"Answer:\n{answer}\n")
#     print("Sources used:")
#     for i, doc in enumerate(chunks):
#         source = doc.metadata.get("filename", "unknown")
#         page = doc.metadata.get("page", "?")
#         paragraph = doc.page_content.strip()
#         print(f"[{i+1}] {source} — page {page}")