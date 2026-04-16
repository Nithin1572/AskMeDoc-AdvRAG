from hybrid_retriever import ask

def main():
    print("\n AskMeDoc — RAG over LLM Research Papers")

    while True:
        query = input("\n Enter your question (or 'exit/quit/clear' to quit): ").strip()

        if query.lower() == "exit" or query.lower() == "quit" or query.lower() == "clear":
            print("\nGoodbye!")
            break

        if not query:
            print("Please enter a valid question.")
            continue

        answer, chunks = ask(query)

        print(f"\nAnswer: {answer}")

        print("Sources used:")
        for i, doc in enumerate(chunks):
            source = doc.metadata.get("filename", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"  [{i+1}] {source} — page {page}")

if __name__ == "__main__":
    main()