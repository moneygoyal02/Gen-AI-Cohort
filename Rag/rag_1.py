from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
import json

file_path = Path(__file__).parent.parent / "pdf_node.pdf"

loader = PyPDFLoader(str(file_path))
docs = loader.load()

# print(docs[0])

# above code is just to load the pdf file 
# pypdfloaer breaks the pdf into pages and returns a list of documents
# each document is a dictionary with page content and metadata  

# now we use text splitter to split the document into smaller chunks because it is too large to process in one go

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(documents=docs)

print(f"Total number of chunks: {len(split_docs)}")
print(f"lenght of document :{len(docs)}")

# now we can use these chunks to create embeddings and store them in a vector store
# we will use openai embeddings for this purpose
# we will use the langchain library to create embeddings and store them in a vector store

from langchain_openai import OpenAIEmbeddings

# function to create embeddings for the split documents

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    
)

# we will save the embeddings in a vector store (quadrant)-(db)

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="rag_collection",   
#     embedding=embedder,
# )

# vector_store.add_documents(documents=split_docs)
# print("embeddings created and stored in vector store")
# above part is commented because we don't want to create embeddings every time we run the code
# we will use the existing embeddings stored in the vector store

# now we will use the vector store to retrieve the relevant documents for a given query

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="rag_collection",
    embedding=embedder,
)

query = input("> ")
relevant_chunks = retriver.similarity_search(
    query=query,
)

# print("relevant chunks retrieved from the vector store:", search_results)


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

