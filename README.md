# AskMeDoc-AdvRAG

A production-grade Retrieval Augmented Generation (RAG) system built on top of local LLMs — no API keys, no rate limits, runs entirely offline.

> 📖 For a detailed explanation of this project, check out the [Medium blog post](https://medium.com/@nithintpr1572/test-blog-5fa194f853cf)

---

## Tech Stack
- **LLM & Inference:** Gemma 3:1B, Gemma 3:4b via Ollama (local, offline)
- **Embeddings:** nomic-embed-text via Ollama
- **Vector Store:** ChromaDB
- **Retrieval & Ranking:** rank_bm25, RRF, Cross-Encoder
- **Framework & Libraries:** Langchain, PyPDF, TikToken
- **Evaluation:** RAGAS (Faithfulness, Answer Relevance)
- **DevOps & Workflow:** Git, GitHub Actions

---

## Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed on your system
- Git installed

## Steps
1. Clone the repository
> `git clone https://github.com/Nithin1572/AskMeDoc-AdvRAG.git`
> `cd AskMeDoc-AdvRAG`
2. Create and activate a virtual environment
> `python3 -m venv venv`
> `source venv/bin/activate`
3. Install dependencies from `requirements.txt`
> `pip install -r requirements.txt`
4. Pull the required Ollama models
>`ollama pull nomic-embed-text`
5. Add your research paper PDFs to the `docs/` folder
6. Run `src/ingest.py` to load, chunk and embed documents
7. Run the Application `src/main.py`.