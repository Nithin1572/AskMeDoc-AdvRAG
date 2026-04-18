import sys
sys.path.append(".")

from src.retrieval import load_vectorstore, retrieve_chunks
import json

with open("eval_dataset.json", "r") as f:
    dataset = json.load(f)

vectorstore = load_vectorstore()

for item in dataset:
    query = item["question"]
    chunks = retrieve_chunks(query, vectorstore)
    if chunks:
        page = chunks[0].metadata.get("page", "?")
        item["page"] = page

with open("eval_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Done! Page numbers added to eval_dataset.json")