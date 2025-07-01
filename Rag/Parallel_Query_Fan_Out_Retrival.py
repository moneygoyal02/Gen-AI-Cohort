from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
from pathlib import Path
import json

# Load environment variables (for OpenAI API keys etc.)
load_dotenv()

# ---------- STEP 1: Load and Chunk the PDF ----------
file_path = Path(__file__).parent.parent / "pdf_node.pdf"
loader = PyPDFLoader(str(file_path))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
)
split_docs = text_splitter.split_documents(docs)

print(f"Total number of chunks: {len(split_docs)}")
print(f"Number of original pages: {len(docs)}")

# ---------- STEP 2: Embeddings Setup ----------
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# ---------- STEP 3: Vector Store ----------
# Uncomment this only if you're creating the vector store for the first time
# vector_store = QdrantVectorStore.from_documents(
#     documents=split_docs,
#     url="http://localhost:6333",
#     collection_name="rag_collection",
#     embedding=embedder,
# )
# print("Embeddings created and stored in vector store")

# Use existing vector store
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="rag_collection",
    embedding=embedder,
).as_retriever()

# ---------- STEP 4: MultiQueryRetriever ----------
llm = ChatOpenAI(temperature=0)
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
    include_original=True  # include original query with the rewrites
)

# ---------- STEP 5: Get Query from User ----------
query = input("> ")

# ---------- STEP 6: Run Fan-Out Retrieval ----------

print("\n🔍 Generating multiple rewritten queries...")

# This runs the multi-query retriever and gets the chunks
relevant_chunks = multi_retriever.invoke(query)

# Show how many total chunks were retrieved
print(f"\n✅ Total relevant chunks retrieved: {len(relevant_chunks)}\n")

# Preview first few chunks (for debugging)
print("🧩 Preview of some retrieved chunks:\n")
for i, chunk in enumerate(relevant_chunks[:5], 1):
    print(f"Chunk {i}:")
    print(f"  - Page content (preview): {chunk.page_content[:200]}...\n")


# ---------- STEP 7: Define Prompt Template ----------
SYSTEM_PROMPT = f"""
You are an AI assistant who responds to user queries based on the provided context.

Context:
{relevant_chunks}

Instructions:
1. Follow these steps in order: "analyse", "think", "output", "validate", "result".
2. Think at least 5-6 steps before giving final result.
3. Return responses strictly in JSON format: {{ "step": "...", "content": "..." }}

Output Format:
{{ "step": "string", "content": "string" }}

Example:
Input: What is 2 + 2.
Output: {{ "step": "analyse", "content": "Alright! The user is asking a basic arithmetic question." }}
Output: {{ "step": "think", "content": "To perform addition I must go from left to right and add operands." }}
...
"""

from openai import OpenAI
client = OpenAI()

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": query},
]

# ---------- STEP 8: Chat Completion Loop ----------
while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=messages
    )

    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({"role": "assistant", "content": json.dumps(parsed_response)})

    print(f"🧠 Step ({parsed_response['step']}): {parsed_response['content']}")

    if parsed_response.get("step") == "result":
        break


# Total number of chunks: 266
# Number of original pages: 125
# > How does Node.js manage concurrency in a single-threaded model?

# 🔍 Generating multiple rewritten queries...

# ✅ Total relevant chunks retrieved: 5

# 🧩 Preview of some retrieved chunks:

# Chunk 1:
#   - Page content (preview): Section 6: Asynchronous Node.js ........................................................................................ 26 
# Lesson 1: Section Intro ......................................................

# Chunk 2:
#   - Page content (preview): Version 1.0 27 
# $ node app.js 
# Starting 
# Stopping 
# 2 Second Timer 
# Notice that “Stopping” prints before “2 Second Timer”. That’s because setTimeout is 
# asynchronous and non-blocking. The setTimeout ca...

# Chunk 3:
#   - Page content (preview): Version 1.0 26 
# Section 6: Asynchronous Node.js
# Lesson 1: Section Intro
# It’s time to connect your application with the outside world. In this section, you’ll explore
# the asynchronous nature of Node...

# Chunk 4:
#   - Page content (preview): Version 1.0 10
# Section 3: Node.js Module System
# Lesson 1: Section Intro
# The best way to get started with Node.js is to explore its module system. The module
# system lets you load external libraries...

# Chunk 5:
#   - Page content (preview): Version 1.0 31
# Callback functions are at the core of asynchronous development. When you perform an
# asynchronous operation, you’ll provide Node with a callback function. Node will then call
# the call...

# 🧠 Step (analyse): The user wants to understand how Node.js handles multiple simultaneous actions given that it operates on a single-threaded model.
# 🧠 Step (think): Node.js uses an event-driven and asynchronous model to handle concurrency despite being single-threaded.
# 🧠 Step (think): Key components of managing concurrency in Node.js include the call stack, callback queue, and event loop.
# 🧠 Step (think): Asynchronous operations, such as HTTP requests or file operations, do not block the single main thread.
# 🧠 Step (think): These operations use system resources (like OS-level threads) or library support which allows Node.js to offload work while the main thread continues processing.
# 🧠 Step (think): Once an asynchronous operation is complete, callbacks are pushed to the callback queue, from where the event loop moves them to the call stack when it is empty.
# 🧠 Step (validate): Each asynchronous operation is managed by a combination of the system resources available and Node.js's event-driven architecture, allowing efficient handling of multiple tasks.
# 🧠 Step (result): Node.js manages concurrency through its event-driven and non-blocking architecture, using the call stack, callback queue, and event loop to efficiently handle multiple asynchronous operations within a single-threaded environment.