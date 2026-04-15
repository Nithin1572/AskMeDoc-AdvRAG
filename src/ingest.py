import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

def load_pdfs(docs_folder = "docs"):
    all_text = []
    for filename in os.listdir(docs_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(docs_folder, filename)
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            all_text.append({
                "filename": filename,
                "content": text
            })
            print(f"Loaded {filename} ({len(reader.pages)} pages)")
    return all_text

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    all_chunks = []

    for doc in documents:
        chunks = splitter.split_text(doc["content"])
        for chunk in chunks:
            all_chunks.append(Document(
                page_content=chunk,
                metadata={"filename": doc["filename"]}
            ))

    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks
    
def store_embeddings(chunks):
    print("\nGenerating embeddings and storing in ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

if __name__ == "__main__":
    documents = load_pdfs()
    print(f"\nTotal documents loaded: {len(documents)}")
    
    chunks = chunk_documents(documents)
    
    vectorstore = store_embeddings(chunks)