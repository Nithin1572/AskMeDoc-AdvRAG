from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
import yaml

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
    with open("prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    context = ""
    for i, doc in enumerate(chunks):
        source = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "?")
        context += f"\n[{i+1}] (Source: {source}, Page: {page})\n{doc.page_content}\n"

    prompt = prompts["rag_prompt"].format(
        context=context,
        query=query,
        num_chunks=len(chunks)
    )
    return prompt

def ask(query):
    vectorstore = load_vectorstore()
    chunks = retrieve_chunks(query, vectorstore)
    prompt = build_prompt(query, chunks)
    
    llm = OllamaLLM(model=LLM_MODEL)
    answer = llm.invoke(prompt)
    
    return answer, chunks