import chromadb
import ollama

# Set up a local, persistent memory store
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="agent_memory")

# Step 1: Store some example memories
memories = [
    "The user's favorite programming language is Python.",
    "The user is building a local AI agent on an RTX 5060 Ti.",
    "The user prefers PowerShell over CMD for development.",
]

for i, memory in enumerate(memories):
    collection.add(
        documents=[memory],
        ids=[f"memory_{i}"]
    )

print("Memories stored.\n")

# Step 2: Ask a question, retrieve relevant memory
query = "What GPU is the user using?"
results = collection.query(
    query_texts=[query],
    n_results=1
)

retrieved_memory = results["documents"][0][0]
print(f"Retrieved memory: {retrieved_memory}\n")

# Step 3: Feed the retrieved memory + question into Qwen
prompt = f"""Using this context: "{retrieved_memory}"

Answer the question: {query}"""

response = ollama.generate(model="qwen2.5:14b", prompt=prompt)
print("Qwen's answer:")
print(response["response"])