from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import re

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
from openai import OpenAI

# Initialize components
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="rag_collection",
    embedding=embedder,
)
client = OpenAI()

def generate_step_back_questions(query: str, openai_client) -> Dict[str, Any]:
    """
    Generate step-back (abstraction) questions from the original query.
    """
    step_back_prompt = f"""
    You are an expert at creating high-level, conceptual questions from specific queries.
    
    Original Query: "{query}"
    
    Your task: Create 2-3 step-back questions that are more general and conceptual than the original query.
    These questions should help understand the broader context and principles.
    
    Rules:
    1. Step-back questions should be more abstract and general
    2. They should cover fundamental concepts related to the query
    3. They should help build foundational understanding
    
    Return your response in JSON format:
    {{
        "original_query": "...",
        "step_back_questions": [
            "question1",
            "question2", 
            "question3"
        ],
        "reasoning": "Why these step-back questions are helpful..."
    }}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": step_back_prompt}]
    )
    
    return json.loads(response.choices[0].message.content)

def generate_chain_of_thought_decomposition(query: str, openai_client) -> Dict[str, Any]:
    """
    Break down the query into logical sub-questions using Chain-of-Thought reasoning.
    """
    cot_prompt = f"""
    You are an expert at breaking down complex questions into logical sub-questions.
    
    Original Query: "{query}"
    
    Think step by step about what information is needed to fully answer this query:
    
    1. First, identify the main components of the question
    2. Then, think about what background knowledge is needed
    3. Finally, break it into 3-5 specific sub-questions that, when answered together, would fully address the original query
    
    Return your response in JSON format:
    {{
        "original_query": "...",
        "reasoning_steps": [
            "Step 1: Analysis of main components...",
            "Step 2: Identification of background knowledge needed...",
            "Step 3: Breaking into sub-questions..."
        ],
        "sub_questions": [
            "sub_question_1",
            "sub_question_2",
            "sub_question_3",
            "sub_question_4",
            "sub_question_5"
        ],
        "question_relationships": "How these sub-questions connect to answer the main query..."
    }}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": cot_prompt}]
    )
    
    return json.loads(response.choices[0].message.content)

def decompose_query(query: str, openai_client) -> Dict[str, Any]:
    """
    Complete query decomposition using both Step-Back and Chain-of-Thought approaches.
    """
    print("🔄 Performing Step-Back Prompting...")
    step_back_result = generate_step_back_questions(query, openai_client)
    
    print("🧠 Performing Chain-of-Thought Decomposition...")
    cot_result = generate_chain_of_thought_decomposition(query, openai_client)
    
    # Combine both approaches
    decomposition_result = {
        "original_query": query,
        "step_back_analysis": step_back_result,
        "chain_of_thought_analysis": cot_result,
        "all_queries": [query] + step_back_result["step_back_questions"] + cot_result["sub_questions"]
    }
    
    return decomposition_result

def reciprocal_rank_fusion(results_list: List[List[Any]], k: int = 60) -> List[Any]:
    """
    Implement Reciprocal Rank Fusion to combine multiple ranked lists.
    """
    rrf_scores = defaultdict(float)
    doc_objects = {}
    
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.page_content[:100]
            doc_objects[doc_id] = doc
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_objects[doc_id] for doc_id, score in sorted_docs]

def retrieve_with_decomposed_queries(decomposed_queries: Dict[str, Any], retriever, top_k: int = 15) -> Tuple[List[Any], List[str]]:
    """
    Retrieve documents using decomposed queries and combine with RRF.
    """
    all_results = []
    query_types = []
    
    # Retrieve for original query (highest weight)
    print("🔍 Retrieving for original query...")
    original_results = retriever.similarity_search(
        query=decomposed_queries["original_query"],
        k=top_k
    )
    all_results.append(original_results)
    query_types.append("original")
    
    # Retrieve for step-back questions (medium weight)
    print("🔍 Retrieving for step-back questions...")
    for i, step_back_q in enumerate(decomposed_queries["step_back_analysis"]["step_back_questions"]):
        results = retriever.similarity_search(query=step_back_q, k=top_k//2)
        all_results.append(results)
        query_types.append(f"step_back_{i}")
    
    # Retrieve for sub-questions (lower weight but comprehensive)
    print("🔍 Retrieving for chain-of-thought sub-questions...")
    for i, sub_q in enumerate(decomposed_queries["chain_of_thought_analysis"]["sub_questions"]):
        results = retriever.similarity_search(query=sub_q, k=top_k//3)
        all_results.append(results)
        query_types.append(f"sub_question_{i}")
    
    # Apply RRF to combine all results
    print("🔄 Applying Reciprocal Rank Fusion...")
    fused_results = reciprocal_rank_fusion(all_results, k=60)
    
    return fused_results[:top_k], query_types

def create_enhanced_context(decomposed_queries: Dict[str, Any], retrieved_docs: List[Any]) -> str:
    """
    Create enhanced context that includes decomposition reasoning.
    """
    context = f"""
QUERY DECOMPOSITION ANALYSIS:

Original Query: {decomposed_queries['original_query']}

STEP-BACK QUESTIONS (High-level concepts):
{chr(10).join([f"- {q}" for q in decomposed_queries['step_back_analysis']['step_back_questions']])}

CHAIN-OF-THOUGHT SUB-QUESTIONS (Detailed breakdown):
{chr(10).join([f"- {q}" for q in decomposed_queries['chain_of_thought_analysis']['sub_questions']])}

REASONING FOR DECOMPOSITION:
Step-Back Reasoning: {decomposed_queries['step_back_analysis']['reasoning']}
CoT Reasoning: {decomposed_queries['chain_of_thought_analysis']['question_relationships']}

RETRIEVED CONTEXT (RRF-ranked based on all decomposed queries):
{retrieved_docs}
"""
    return context

def generate_decomposition_aware_response(original_query: str, context: str, decomposition: Dict[str, Any], openai_client) -> None:
    """
    Generate response using decomposition-aware system prompt.
    """
    DECOMPOSITION_SYSTEM_PROMPT = f"""
You are an advanced AI assistant that excels at comprehensive reasoning using query decomposition.

You have been provided with:
1. An original query that has been decomposed using Step-Back Prompting and Chain-of-Thought reasoning
2. Context retrieved using multiple related queries for maximum coverage
3. Both high-level conceptual information and detailed specific information

CONTEXT AND DECOMPOSITION:
{context}

INSTRUCTIONS:
1. Follow these steps: "analyze_decomposition", "synthesize_step_back", "integrate_cot", "validate_completeness", "result"
2. In your analysis, consider:
   - How the step-back questions provide foundational understanding
   - How the sub-questions address specific aspects
   - How all pieces connect to fully answer the original query
3. Provide a comprehensive response that leverages both abstract concepts and specific details
4. Return responses in JSON format: {{"step": "...", "content": "..."}}

Your response should demonstrate understanding of both the broad concepts (from step-back) and specific details (from CoT decomposition).
"""
    
    messages = [
        {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
        {"role": "user", "content": original_query},
    ]
    
    step_count = 0
    max_steps = 10  # Prevent infinite loops
    
    while step_count < max_steps:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=messages
        )
        
        parsed_response = json.loads(response.choices[0].message.content)
        messages.append({"role": "assistant", "content": json.dumps(parsed_response)})
        
        print(f"🧠 Step ({parsed_response['step']}): {parsed_response['content']}")
        
        if parsed_response.get("step") == "result":
            break
            
        step_count += 1

def process_query_with_decomposition(query: str, retriever, openai_client) -> None:
    """
    Main function to process query using decomposition-enhanced RAG pipeline.
    """
    print("=" * 80)
    print("🚀 STARTING DECOMPOSED RAG PROCESSING")
    print("=" * 80)
    
    # Step 1: Decompose the query
    print("\n📋 STEP 1: QUERY DECOMPOSITION")
    decomposed = decompose_query(query, openai_client)
    
    print(f"\n🎯 Original Query: {query}")
    print(f"📚 Step-back questions: {len(decomposed['step_back_analysis']['step_back_questions'])}")
    print(f"🧩 Sub-questions: {len(decomposed['chain_of_thought_analysis']['sub_questions'])}")
    print(f"🔍 Total queries for retrieval: {len(decomposed['all_queries'])}")
    
    # Step 2: Enhanced retrieval
    print("\n📋 STEP 2: ENHANCED RETRIEVAL WITH DECOMPOSITION")
    retrieved_docs, query_types = retrieve_with_decomposed_queries(decomposed, retriever, top_k=10)
    
    # Step 3: Create enhanced context
    enhanced_context = create_enhanced_context(decomposed, retrieved_docs)
    
    # Step 4: Generate response with decomposition-aware prompt
    print("\n📋 STEP 3: GENERATING DECOMPOSITION-AWARE RESPONSE")
    generate_decomposition_aware_response(query, enhanced_context, decomposed, openai_client)
    
    print("\n" + "=" * 80)
    print("✅ DECOMPOSED RAG PROCESSING COMPLETE")
    print("=" * 80)

def display_decomposition_summary(decomposed: Dict[str, Any]) -> None:
    """
    Display a summary of the query decomposition for debugging/analysis.
    """
    print("\n" + "="*60)
    print("📊 QUERY DECOMPOSITION SUMMARY")
    print("="*60)
    
    print(f"🎯 Original Query:")
    print(f"   {decomposed['original_query']}")
    
    print(f"\n📚 Step-Back Questions:")
    for i, q in enumerate(decomposed['step_back_analysis']['step_back_questions'], 1):
        print(f"   {i}. {q}")
    
    print(f"\n🧩 Chain-of-Thought Sub-Questions:")
    for i, q in enumerate(decomposed['chain_of_thought_analysis']['sub_questions'], 1):
        print(f"   {i}. {q}")
    
    print(f"\n💡 Step-Back Reasoning:")
    print(f"   {decomposed['step_back_analysis']['reasoning']}")
    
    print(f"\n🔗 CoT Relationships:")
    print(f"   {decomposed['chain_of_thought_analysis']['question_relationships']}")
    
    print("="*60)

def main():
    """
    Main execution function
    """
    # Get user query
    query = input("🤖 Enter your query > ")
    
    # Optional: Show detailed decomposition summary
    show_summary = input("Show decomposition summary? (y/n): ").lower().strip() == 'y'
    
    if show_summary:
        # First decompose to show summary
        decomposed = decompose_query(query, client) 
        display_decomposition_summary(decomposed)
        
        # Ask if user wants to continue
        if input("\nContinue with full processing? (y/n): ").lower().strip() != 'y':
            return
    
    # Process the query using decomposition-enhanced RAG
    process_query_with_decomposition(query, retriever, client)

# Main execution
if __name__ == "__main__":
    main()













# from langchain_community.document_loaders import PyPDFLoader
# from pathlib import Path
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
# load_dotenv()
# import json
# import numpy as np
# from typing import List, Dict, Any, Tuple
# from collections import defaultdict

# # Load and split documents
# file_path = Path(__file__).parent.parent / "pdf_node.pdf"
# loader = PyPDFLoader(str(file_path))
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, 
#     chunk_overlap=200,
# )
# split_docs = text_splitter.split_documents(documents=docs)
# print(f"Total number of chunks: {len(split_docs)}")
# print(f"Length of document: {len(docs)}")

# from langchain_openai import OpenAIEmbeddings
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from openai import OpenAI

# # Initialize components
# embedder = OpenAIEmbeddings(model="text-embedding-3-large")
# retriever = QdrantVectorStore.from_existing_collection(
#     url="http://localhost:6333",
#     collection_name="rag_collection",
#     embedding=embedder,
# )
# client = OpenAI()

# class QueryDecomposer:
#     """
#     Implements query decomposition using Step-Back Prompting and Chain-of-Thought reasoning.
#     """
    
#     def __init__(self, openai_client):
#         self.client = openai_client
    
#     def step_back_prompting(self, query: str) -> Dict[str, Any]:
#         """
#         Generate step-back (abstraction) questions from the original query.
#         """
#         step_back_prompt = f"""
#         You are an expert at creating high-level, conceptual questions from specific queries.
        
#         Original Query: "{query}"
        
#         Your task: Create 2-3 step-back questions that are more general and conceptual than the original query.
#         These questions should help understand the broader context and principles.
        
#         Rules:
#         1. Step-back questions should be more abstract and general
#         2. They should cover fundamental concepts related to the query
#         3. They should help build foundational understanding
        
#         Return your response in JSON format:
#         {{
#             "original_query": "...",
#             "step_back_questions": [
#                 "question1",
#                 "question2", 
#                 "question3"
#             ],
#             "reasoning": "Why these step-back questions are helpful..."
#         }}
#         """
        
#         response = self.client.chat.completions.create(
#             model="gpt-4o",
#             response_format={"type": "json_object"},
#             messages=[{"role": "user", "content": step_back_prompt}]
#         )
        
#         return json.loads(response.choices[0].message.content)
    
#     def chain_of_thought_decomposition(self, query: str) -> Dict[str, Any]:
#         """
#         Break down the query into logical sub-questions using Chain-of-Thought reasoning.
#         """
#         cot_prompt = f"""
#         You are an expert at breaking down complex questions into logical sub-questions.
        
#         Original Query: "{query}"
        
#         Think step by step about what information is needed to fully answer this query:
        
#         1. First, identify the main components of the question
#         2. Then, think about what background knowledge is needed
#         3. Finally, break it into 3-5 specific sub-questions that, when answered together, would fully address the original query
        
#         Return your response in JSON format:
#         {{
#             "original_query": "...",
#             "reasoning_steps": [
#                 "Step 1: Analysis of main components...",
#                 "Step 2: Identification of background knowledge needed...",
#                 "Step 3: Breaking into sub-questions..."
#             ],
#             "sub_questions": [
#                 "sub_question_1",
#                 "sub_question_2",
#                 "sub_question_3",
#                 "sub_question_4",
#                 "sub_question_5"
#             ],
#             "question_relationships": "How these sub-questions connect to answer the main query..."
#         }}
#         """
        
#         response = self.client.chat.completions.create(
#             model="gpt-4o",
#             response_format={"type": "json_object"},
#             messages=[{"role": "user", "content": cot_prompt}]
#         )
        
#         return json.loads(response.choices[0].message.content)
    
#     def decompose_query(self, query: str) -> Dict[str, Any]:
#         """
#         Complete query decomposition using both Step-Back and Chain-of-Thought approaches.
#         """
#         print("🔄 Performing Step-Back Prompting...")
#         step_back_result = self.step_back_prompting(query)
        
#         print("🧠 Performing Chain-of-Thought Decomposition...")
#         cot_result = self.chain_of_thought_decomposition(query)
        
#         # Combine both approaches
#         decomposition_result = {
#             "original_query": query,
#             "step_back_analysis": step_back_result,
#             "chain_of_thought_analysis": cot_result,
#             "all_queries": [query] + step_back_result["step_back_questions"] + cot_result["sub_questions"]
#         }
        
#         return decomposition_result

# def reciprocal_rank_fusion(results_list: List[List[Any]], k: int = 60) -> List[Any]:
#     """RRF implementation from previous version"""
#     rrf_scores = defaultdict(float)
#     doc_objects = {}
    
#     for results in results_list:
#         for rank, doc in enumerate(results):
#             doc_id = doc.page_content[:100]
#             doc_objects[doc_id] = doc
#             rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    
#     sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
#     return [doc_objects[doc_id] for doc_id, score in sorted_docs]

# def enhanced_retrieval_with_decomposition(decomposed_queries: Dict[str, Any], top_k: int = 15) -> List[Any]:
#     """
#     Retrieve documents using decomposed queries and combine with RRF.
#     """
#     all_results = []
#     query_types = []
    
#     # Retrieve for original query (highest weight)
#     print("🔍 Retrieving for original query...")
#     original_results = retriever.similarity_search(
#         query=decomposed_queries["original_query"],
#         k=top_k
#     )
#     all_results.append(original_results)
#     query_types.append("original")
    
#     # Retrieve for step-back questions (medium weight)
#     print("🔍 Retrieving for step-back questions...")
#     for i, step_back_q in enumerate(decomposed_queries["step_back_analysis"]["step_back_questions"]):
#         results = retriever.similarity_search(query=step_back_q, k=top_k//2)
#         all_results.append(results)
#         query_types.append(f"step_back_{i}")
    
#     # Retrieve for sub-questions (lower weight but comprehensive)
#     print("🔍 Retrieving for chain-of-thought sub-questions...")
#     for i, sub_q in enumerate(decomposed_queries["chain_of_thought_analysis"]["sub_questions"]):
#         results = retriever.similarity_search(query=sub_q, k=top_k//3)
#         all_results.append(results)
#         query_types.append(f"sub_question_{i}")
    
#     # Apply RRF to combine all results
#     print("🔄 Applying Reciprocal Rank Fusion...")
#     fused_results = reciprocal_rank_fusion(all_results, k=60)
    
#     return fused_results[:top_k], query_types

# def create_enhanced_context(decomposed_queries: Dict[str, Any], retrieved_docs: List[Any]) -> str:
#     """
#     Create enhanced context that includes decomposition reasoning.
#     """
#     context = f"""
# QUERY DECOMPOSITION ANALYSIS:

# Original Query: {decomposed_queries['original_query']}

# STEP-BACK QUESTIONS (High-level concepts):
# {chr(10).join([f"- {q}" for q in decomposed_queries['step_back_analysis']['step_back_questions']])}

# CHAIN-OF-THOUGHT SUB-QUESTIONS (Detailed breakdown):
# {chr(10).join([f"- {q}" for q in decomposed_queries['chain_of_thought_analysis']['sub_questions']])}

# REASONING FOR DECOMPOSITION:
# Step-Back Reasoning: {decomposed_queries['step_back_analysis']['reasoning']}
# CoT Reasoning: {decomposed_queries['chain_of_thought_analysis']['question_relationships']}

# RETRIEVED CONTEXT (RRF-ranked based on all decomposed queries):
# {retrieved_docs}
# """
#     return context

# class DecomposedRAGProcessor:
#     """
#     Main RAG processor that uses query decomposition for enhanced retrieval and reasoning.
#     """
    
#     def __init__(self, openai_client):
#         self.client = openai_client
#         self.query_decomposer = QueryDecomposer(openai_client)
    
#     def process_query(self, query: str) -> None:
#         """
#         Process query using decomposition-enhanced RAG pipeline.
#         """
#         print("=" * 80)
#         print("🚀 STARTING DECOMPOSED RAG PROCESSING")
#         print("=" * 80)
        
#         # Step 1: Decompose the query
#         print("\n📋 STEP 1: QUERY DECOMPOSITION")
#         decomposed = self.query_decomposer.decompose_query(query)
        
#         print(f"\n🎯 Original Query: {query}")
#         print(f"📚 Step-back questions: {len(decomposed['step_back_analysis']['step_back_questions'])}")
#         print(f"🧩 Sub-questions: {len(decomposed['chain_of_thought_analysis']['sub_questions'])}")
#         print(f"🔍 Total queries for retrieval: {len(decomposed['all_queries'])}")
        
#         # Step 2: Enhanced retrieval
#         print("\n📋 STEP 2: ENHANCED RETRIEVAL WITH DECOMPOSITION")
#         retrieved_docs, query_types = enhanced_retrieval_with_decomposition(decomposed, top_k=10)
        
#         # Step 3: Create enhanced context
#         enhanced_context = create_enhanced_context(decomposed, retrieved_docs)
        
#         # Step 4: Generate response with decomposition-aware prompt
#         print("\n📋 STEP 3: GENERATING DECOMPOSITION-AWARE RESPONSE")
#         self.generate_response(query, enhanced_context, decomposed)
    
#     def generate_response(self, original_query: str, context: str, decomposition: Dict[str, Any]) -> None:
#         """
#         Generate response using decomposition-aware system prompt.
#         """
#         DECOMPOSITION_SYSTEM_PROMPT = f"""
# You are an advanced AI assistant that excels at comprehensive reasoning using query decomposition.

# You have been provided with:
# 1. An original query that has been decomposed using Step-Back Prompting and Chain-of-Thought reasoning
# 2. Context retrieved using multiple related queries for maximum coverage
# 3. Both high-level conceptual information and detailed specific information

# CONTEXT AND DECOMPOSITION:
# {context}

# INSTRUCTIONS:
# 1. Follow these steps: "analyze_decomposition", "synthesize_step_back", "integrate_cot", "validate_completeness", "result"
# 2. In your analysis, consider:
#    - How the step-back questions provide foundational understanding
#    - How the sub-questions address specific aspects
#    - How all pieces connect to fully answer the original query
# 3. Provide a comprehensive response that leverages both abstract concepts and specific details
# 4. Return responses in JSON format: {{"step": "...", "content": "..."}}

# Your response should demonstrate understanding of both the broad concepts (from step-back) and specific details (from CoT decomposition).
# """
        
#         messages = [
#             {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
#             {"role": "user", "content": original_query},
#         ]
        
#         step_count = 0
#         max_steps = 10  # Prevent infinite loops
        
#         while step_count < max_steps:
#             response = self.client.chat.completions.create(
#                 model="gpt-4o",
#                 response_format={"type": "json_object"},
#                 messages=messages
#             )
            
#             parsed_response = json.loads(response.choices[0].message.content)
#             messages.append({"role": "assistant", "content": json.dumps(parsed_response)})
            
#             print(f"🧠 Step ({parsed_response['step']}): {parsed_response['content']}")
            
#             if parsed_response.get("step") == "result":
#                 break
                
#             step_count += 1
        
#         print("\n" + "=" * 80)
#         print("✅ DECOMPOSED RAG PROCESSING COMPLETE")
#         print("=" * 80)

# # Main execution
# if __name__ == "__main__":
#     # Initialize the decomposed RAG processor
#     rag_processor = DecomposedRAGProcessor(client)
    
#     # Get user query
#     query = input("🤖 Enter your query > ")
    
#     # Process the query using decomposition-enhanced RAG
#     rag_processor.process_query(query)