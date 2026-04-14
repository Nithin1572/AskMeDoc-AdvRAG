from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma3:1b")
response = llm.invoke("Say hello in one sentence.")
print(response.content)