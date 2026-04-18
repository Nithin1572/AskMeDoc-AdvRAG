from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from datasets import Dataset
from src.hybrid_retriever import ask
from src.main import load_model
import json
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

LLM_MODEL   = config["llm_model"]
EMBED_MODEL = config["embed_model"]

llm = LangchainLLMWrapper(OllamaLLM(model=LLM_MODEL))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))

with open("eval_dataset.json", "r") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} questions from eval_dataset.json")

questions = []
answers = []
contexts = []
ground_truths = []

print("Running questions through RAG pipeline...")
model = load_model()

for i, item in enumerate(dataset[:5]):
    print(f"  [{i+1}/{len(dataset)}] {item['question'][:60]}...")
    answer, chunks = ask(item["question"], model)
    questions.append(item["question"])
    answers.append(answer)
    contexts.append([doc.page_content for doc in chunks])
    ground_truths.append(item["answer"])

print(f"\nDone! Collected {len(answers)} answers.")

print("\nRunning RAGAS evaluation...")

ragas_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

faithfulness.llm = llm
faithfulness.embeddings = embeddings
answer_relevancy.llm = llm
answer_relevancy.embeddings = embeddings

results = evaluate(
    ragas_dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=llm,
    embeddings=embeddings,
    raise_exceptions=False,
)

print("\n── Evaluation Results ────────────────────────────────")
faithfulness_score = results['faithfulness']
relevancy_score = results['answer_relevancy']

if isinstance(faithfulness_score, list):
    faithfulness_score = sum(x for x in faithfulness_score if x is not None) / max(len([x for x in faithfulness_score if x is not None]), 1)
if isinstance(relevancy_score, list):
    relevancy_score = sum(x for x in relevancy_score if x is not None) / max(len([x for x in relevancy_score if x is not None]), 1)

print(f"  Faithfulness:     {faithfulness_score:.4f}")
print(f"  Answer Relevancy: {relevancy_score:.4f}")

FAITHFULNESS_THRESHOLD = 0.80
RELEVANCY_THRESHOLD = 0.25

print("\n── Threshold Check ───────────────────────────────────")
passed = True

if faithfulness_score < FAITHFULNESS_THRESHOLD:
    print(f"  ❌ Faithfulness {faithfulness_score:.4f} below threshold {FAITHFULNESS_THRESHOLD}")
    passed = False
else:
    print(f"  ✅ Faithfulness {faithfulness_score:.4f} passed threshold {FAITHFULNESS_THRESHOLD}")

if relevancy_score < RELEVANCY_THRESHOLD:
    print(f"  ❌ Answer Relevancy {relevancy_score:.4f} below threshold {RELEVANCY_THRESHOLD}")
    passed = False
else:
    print(f"  ✅ Answer Relevancy {relevancy_score:.4f} passed threshold {RELEVANCY_THRESHOLD}")

print("─────────────────────────────────────────────────────")

if not passed:
    exit(1)

