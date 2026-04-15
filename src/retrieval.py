from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_DIR  = os.getenv("CHROMA_DIR", "chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL   = os.getenv("LLM_MODEL",  "gemma3:1b")
TOP_K       = int(os.getenv("TOP_K", "5"))

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
        context += f"\n[{i+1}] (Source: {source})\n{doc.page_content}\n"
    
    prompt = f"""You are a helpful research assistant. Answer the question based ONLY on the context provided below. If the answer 
    is not found in the context, say "I don't know based on the provided documents." Do NOT use any outside knowledge. For each point 
    you make, you MUST mention the source number like [1], [2] etc. from the context. Context: {context} Question: {query} Answer:"""
    
    return prompt

def ask(query):
    vectorstore = load_vectorstore()
    chunks = retrieve_chunks(query, vectorstore)
    prompt = build_prompt(query, chunks)
    
    llm = OllamaLLM(model=LLM_MODEL)
    answer = llm.invoke(prompt)
    
    return answer, chunks

if __name__ == "__main__":
    query = "What is the transformer architecture?"
    # query = "what is 1+1?"
    
    print(f"\nQuestion: {query}\n")
    answer, chunks = ask(query)
    
    print(f"Answer:\n{answer}\n")
    print("Sources used:")
    for i, doc in enumerate(chunks):
        source = doc.metadata.get("filename", "unknown")
        print(f"  [{i+1}] {source}")