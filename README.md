# AskMeDoc-AdvRAG

A production-grade Retrieval Augmented Generation (RAG) system built on top of local LLMs — no API keys, no rate limits, runs entirely offline.

> 📖 For a detailed explanation of this project, check out the [Medium blog post](#) *(link coming soon)*

---

## Tech Stack
- **LLM:** Gemma 3 1B via Ollama (local, offline)
- **Embeddings:** nomic-embed-text via Ollama
- **Vector Store:** ChromaDB
- **Framework:** LangChain
- **Language:** Python 3.9+

---

## Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed on your system

## Steps
1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies from `requirements.txt`
4. Pull the required Ollama models
   - `ollama pull nomic-embed-text`
5. Add your research paper PDFs to the `docs/` folder
6. Run `src/ingest.py` to load, chunk and embed documents