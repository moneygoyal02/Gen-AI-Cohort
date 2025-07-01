from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
import json
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

# Load and split documents
file_path = Path(__file__).parent.parent / "pdf_node.pdf"
loader = PyPDFLoader(str(file_path))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
)
split_docs = text_splitter.split_documents(documents=docs)
print(f"Total number of chunks: {len(split_docs)}")
print(f"Length of document: {len(docs)}")

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Initialize embeddings
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

# Initialize vector store
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="rag_collection",
    embedding=embedder,
)

def reciprocal_rank_fusion(results_list: List[List[Any]], k: int = 60) -> List[Any]:
    """
    Implement Reciprocal Rank Fusion to combine multiple ranked lists.
    
    Args:
        results_list: List of ranked result lists from different retrieval methods
        k: Constant used in RRF formula (default: 60)
    
    Returns:
        Fused and re-ranked list of results
    """
    # Dictionary to store RRF scores for each document
    rrf_scores = defaultdict(float)
    # Dictionary to store actual document objects
    doc_objects = {}
    
    # Process each ranked list
    for results in results_list:
        for rank, doc in enumerate(results):
            # Use document content as unique identifier
            doc_id = doc.page_content[:100]  # First 100 chars as ID
            doc_objects[doc_id] = doc
            
            # Calculate RRF score: 1 / (k + rank)
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)  # +1 because rank starts from 0
    
    # Sort by RRF score (descending) and return documents
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return the actual document objects in ranked order
    return [doc_objects[doc_id] for doc_id, score in sorted_docs]

def enhanced_retrieval_with_rrf(query: str, top_k: int = 10) -> List[Any]:
    """
    Enhanced retrieval using multiple search strategies combined with RRF.
    
    Args:
        query: Search query
        top_k: Number of top results to return
    
    Returns:
        List of documents ranked using RRF
    """
    all_results = []
    
    # Method 1: Similarity search with default parameters
    results_1 = retriever.similarity_search(
        query=query,
        k=top_k * 2  # Get more results for better fusion
    )
    all_results.append(results_1)
    
    # Method 2: Similarity search with different parameters (if available)
    # You can customize these based on your vector store capabilities
    try:
        results_2 = retriever.similarity_search(
            query=query,
            k=top_k * 2,
            # Add any additional parameters your vector store supports
        )
        all_results.append(results_2)
    except:
        # If additional parameters not supported, use a modified query
        modified_query = f"Related to: {query}"
        results_2 = retriever.similarity_search(
            query=modified_query,
            k=top_k * 2
        )
        all_results.append(results_2)
    
    # Method 3: Search with query expansion (simple keyword addition)
    expanded_query = f"{query} explanation definition meaning"
    results_3 = retriever.similarity_search(
        query=expanded_query,
        k=top_k * 2
    )
    all_results.append(results_3)
    
    # Apply RRF to combine results
    fused_results = reciprocal_rank_fusion(all_results, k=60)
    
    # Return top_k results
    return fused_results[:top_k]

def advanced_rrf_with_weights(results_list: List[List[Any]], 
                             weights: List[float] = None, 
                             k: int = 60) -> List[Any]:
    """
    Advanced RRF implementation with weighted retrieval methods.
    
    Args:
        results_list: List of ranked result lists from different retrieval methods
        weights: List of weights for each retrieval method
        k: Constant used in RRF formula
    
    Returns:
        Weighted RRF fused and re-ranked list of results
    """
    if weights is None:
        weights = [1.0] * len(results_list)
    
    if len(weights) != len(results_list):
        raise ValueError("Number of weights must match number of result lists")
    
    # Dictionary to store weighted RRF scores
    rrf_scores = defaultdict(float)
    doc_objects = {}
    
    # Process each ranked list with its weight
    for weight, results in zip(weights, results_list):
        for rank, doc in enumerate(results):
            doc_id = doc.page_content[:100]
            doc_objects[doc_id] = doc
            
            # Calculate weighted RRF score
            rrf_scores[doc_id] += weight * (1.0 / (k + rank + 1))
    
    # Sort by weighted RRF score and return documents
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_objects[doc_id] for doc_id, score in sorted_docs]

# Main execution
query = input("> ")

# Use enhanced retrieval with RRF
print("🔍 Retrieving with Reciprocal Rank Fusion...")
relevant_chunks = enhanced_retrieval_with_rrf(query, top_k=5)

# Alternative: Use weighted RRF (uncomment to use)
# print("🔍 Retrieving with Weighted RRF...")
# results_1 = retriever.similarity_search(query=query, k=10)
# results_2 = retriever.similarity_search(query=f"Related to: {query}", k=10)
# results_3 = retriever.similarity_search(query=f"{query} explanation", k=10)
# 
# # Assign higher weight to primary similarity search
# relevant_chunks = advanced_rrf_with_weights(
#     [results_1, results_2, results_3], 
#     weights=[0.5, 0.3, 0.2]
# )[:5]

print(f"📊 Number of relevant chunks retrieved: {len(relevant_chunks)}")

# Enhanced system prompt with RRF information
SYSTEM_PROMPT = f"""
You are an AI assistant who responds to user queries based on the provided context.
The context has been retrieved using Reciprocal Rank Fusion (RRF), which combines 
multiple search strategies to provide the most relevant information.

Context:
{relevant_chunks}

Instructions:
1. Follow these steps in order: "analyse", "think", "output", "validate", "result".
2. Think at least 5-6 steps before giving final result.
3. The provided context has been optimally ranked using RRF for maximum relevance.
4. Return responses strictly in JSON format: {{ "step": "...", "content": "..." }}

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

print("🤖 Processing with RRF-enhanced context...")
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

print("\n✅ RRF-enhanced RAG processing complete!")