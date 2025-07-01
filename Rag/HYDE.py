from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
import json

file_path = Path(__file__).parent.parent / "pdf_node.pdf"

loader = PyPDFLoader(str(file_path))
docs = loader.load()

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
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="rag_collection",
    embedding=embedder,
)

from openai import OpenAI
client = OpenAI()

# =============================================================================
# HyDE (Hypothetical Document Embeddings) Implementation
# =============================================================================

def generate_hypothetical_document(query: str, num_hypotheses: int = 3) -> list:
    """
    Generate hypothetical documents that might contain the answer to the query.
    This helps bridge the gap between question style and document style.
    """
    
    hyde_prompt = f"""
    Given the following query, generate {num_hypotheses} different hypothetical document passages that would likely contain the answer to this query.

    Query: {query}

    Instructions:
    1. Write as if you're creating excerpts from authoritative documents
    2. Use technical/academic language appropriate for the domain
    3. Include specific details and terminology that would appear in real documents
    4. Make each hypothesis focus on different aspects or approaches to the topic
    5. Each passage should be 2-3 sentences long
    6. Do NOT actually answer the question - create document-style content that WOULD contain the answer

    Format your response as a JSON array of strings:
    ["hypothesis1", "hypothesis2", "hypothesis3"]
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an expert at generating hypothetical document passages for information retrieval. Return only a JSON object with a 'hypotheses' array."},
            {"role": "user", "content": hyde_prompt}
        ]
    )
    
    result = json.loads(response.choices[0].message.content)
    return result.get("hypotheses", [])

def hyde_retrieval(query: str, retriever, embedder, k: int = 5) -> list:
    """
    Perform HyDE-enhanced retrieval:
    1. Generate hypothetical documents
    2. Embed them and search for similar real documents
    3. Combine and deduplicate results
    4. Return the most relevant chunks
    """
    
    print("🔬 HyDE Step 1: Generating hypothetical documents...")
    hypothetical_docs = generate_hypothetical_document(query)
    
    print(f"📝 Generated {len(hypothetical_docs)} hypothetical documents:")
    for i, doc in enumerate(hypothetical_docs, 1):
        print(f"   {i}. {doc[:100]}...")
    
    print("\n🔍 HyDE Step 2: Searching with hypothetical documents...")
    
    # Search using each hypothetical document
    all_results = []
    for i, hyp_doc in enumerate(hypothetical_docs):
        print(f"   Searching with hypothesis {i+1}...")
        results = retriever.similarity_search(query=hyp_doc, k=k)
        all_results.extend(results)
    
    # Also search with original query for comparison
    print("   Searching with original query...")
    original_results = retriever.similarity_search(query=query, k=k)
    all_results.extend(original_results)
    
    # Deduplicate based on content (simple approach)
    seen_content = set()
    unique_results = []
    
    for doc in all_results:
        content_hash = hash(doc.page_content[:200])  # Use first 200 chars as identifier
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_results.append(doc)
    
    print(f"✅ HyDE Step 3: Found {len(unique_results)} unique relevant chunks")
    
    # Return top k results (you might want to implement more sophisticated ranking here)
    return unique_results[:k]

# Advanced Few-Shot Prompting: Step-Back Abstraction + Chain-of-Thought
ADVANCED_FEW_SHOT_EXAMPLES = """
=== EXAMPLE 1: Technical Definition Query ===
User Query: "What is neural network architecture optimization?"
Context: "Neural networks require careful design of layers, neurons, activation functions, and hyperparameters to achieve optimal performance..."

STEP-BACK ABSTRACTION:
{"step": "analyse", "content": "This is a technical definition query about optimization in machine learning. I need to step back and consider: What is the broader concept of optimization? How does it apply to neural networks? What are the fundamental principles of system design that apply here?"}

DETAILED CHAIN-OF-THOUGHT:
{"step": "think", "content": "Breaking down the query: 1) Neural networks are computational models, 2) Architecture refers to structure and design, 3) Optimization means finding the best configuration. The context mentions layers, neurons, activation functions, and hyperparameters as key components."}

{"step": "think", "content": "From the step-back perspective: This is fundamentally about system design optimization - finding the best structure for a given task. This applies to many fields: engineering systems, biological systems, organizational structures."}

{"step": "think", "content": "Applying chain-of-thought to specifics: The context indicates optimization involves multiple components - architectural elements (layers, neurons), functional elements (activation functions), and control parameters (hyperparameters)."}

{"step": "think", "content": "Connecting abstract to concrete: Just like optimizing any complex system, neural network optimization requires balancing multiple factors - performance, efficiency, complexity, and generalization capability."}

{"step": "output", "content": "Neural network architecture optimization is the process of systematically designing and tuning the structural and functional components of neural networks to achieve optimal performance. This involves selecting the appropriate number and types of layers, determining neuron configurations, choosing activation functions, and fine-tuning hyperparameters to balance model accuracy, computational efficiency, and generalization capability."}

{"step": "validate", "content": "My response captures both the high-level concept (systematic optimization of complex systems) and the specific technical details (layers, neurons, activation functions, hyperparameters) mentioned in the context."}

{"step": "result", "content": "Neural network architecture optimization is the process of systematically designing and tuning the structural and functional components of neural networks to achieve optimal performance. This involves selecting the appropriate number and types of layers, determining neuron configurations, choosing activation functions, and fine-tuning hyperparameters to balance model accuracy, computational efficiency, and generalization capability."}

=== EXAMPLE 2: Process/Methodology Query ===
User Query: "How does data validation work in machine learning pipelines?"
Context: "Data validation involves checking data quality, consistency, and integrity before training. It includes statistical tests, schema validation, and anomaly detection..."

STEP-BACK ABSTRACTION:
{"step": "analyse", "content": "This is asking about a quality assurance process. Stepping back: What is validation in general? It's about verification, ensuring something meets standards or requirements. This applies across domains - manufacturing quality control, software testing, scientific methodology."}

DETAILED CHAIN-OF-THOUGHT:
{"step": "think", "content": "Chain-of-thought breakdown: 1) Data is the input to ML systems, 2) Validation means checking/verifying, 3) Pipelines are sequential processes, 4) The context mentions specific techniques: statistical tests, schema validation, anomaly detection."}

{"step": "think", "content": "From the step-back view: This is fundamentally about risk management and quality assurance - preventing problems before they propagate through the system. Similar to how we validate inputs in any critical system."}

{"step": "think", "content": "Detailed reasoning on the process: The context suggests a multi-layered approach - checking different aspects of data quality (statistical properties, structural compliance, unusual patterns)."}

{"step": "think", "content": "Connecting abstract to concrete: Just like quality control in manufacturing checks multiple attributes (dimensions, materials, functionality), data validation checks multiple data attributes (distribution, format, consistency)."}

{"step": "think", "content": "Process flow reasoning: Validation happens 'before training' according to context, suggesting it's a gate-keeping function that prevents bad data from corrupting the learning process."}

{"step": "output", "content": "Data validation in machine learning pipelines is a systematic quality assurance process that verifies data integrity before model training. It operates as a multi-layered checkpoint system that includes: statistical tests to verify data distributions and properties, schema validation to ensure structural compliance with expected formats, and anomaly detection to identify unusual patterns or outliers. This validation acts as a critical gate-keeping function that prevents corrupted or inconsistent data from degrading model performance."}

{"step": "validate", "content": "My response captures the high-level concept of quality assurance and risk management, while detailing the specific validation techniques mentioned in the context. It explains both the 'what' and the 'why' of the process."}

{"step": "result", "content": "Data validation in machine learning pipelines is a systematic quality assurance process that verifies data integrity before model training. It operates as a multi-layered checkpoint system that includes: statistical tests to verify data distributions and properties, schema validation to ensure structural compliance with expected formats, and anomaly detection to identify unusual patterns or outliers. This validation acts as a critical gate-keeping function that prevents corrupted or inconsistent data from degrading model performance."}

=== EXAMPLE 3: Comparative/Benefits Query ===
User Query: "What are the advantages of distributed computing over centralized systems?"
Context: "Distributed computing offers fault tolerance, scalability, and parallel processing capabilities. Unlike centralized systems, it can continue operating even if individual nodes fail..."

STEP-BACK ABSTRACTION:
{"step": "analyse", "content": "This is a comparative analysis question about system architectures. Stepping back to first principles: What are the fundamental trade-offs in system design? Centralization vs. decentralization is a classic architectural decision that appears in many domains - government, organizations, technology, biology."}

DETAILED CHAIN-OF-THOUGHT:
{"step": "think", "content": "Chain-of-thought for comparison: 1) Distributed = spread across multiple locations/nodes, 2) Centralized = concentrated in one location/system, 3) Context mentions specific advantages: fault tolerance, scalability, parallel processing, and failure resilience."}

{"step": "think", "content": "Step-back reasoning: This is fundamentally about risk distribution and resource utilization. In any system, concentrating resources/functions creates single points of failure but may improve coordination. Distributing them reduces risk but may complicate coordination."}

{"step": "think", "content": "Detailed analysis of each advantage: Fault tolerance means the system can survive component failures. Scalability means it can grow by adding resources. Parallel processing means multiple operations can happen simultaneously."}

{"step": "think", "content": "Abstract principle application: This follows the principle of redundancy and load distribution found in many robust systems - from biological systems (multiple organs) to infrastructure (power grids, transportation networks)."}

{"step": "think", "content": "Connecting to context specifics: The context explicitly contrasts with centralized systems and mentions the key benefit of continuing operation despite node failures."}

{"step": "output", "content": "Distributed computing offers several key advantages over centralized systems by applying principles of redundancy and load distribution: 1) Fault tolerance - the system remains operational even when individual components fail, eliminating single points of failure, 2) Scalability - computational capacity can be expanded by adding more nodes rather than upgrading a single powerful machine, 3) Parallel processing - multiple tasks can be executed simultaneously across different nodes, significantly improving performance for complex workloads. Unlike centralized systems that create vulnerability through concentration, distributed architectures spread risk and resources across multiple nodes."}

{"step": "validate", "content": "My response successfully combines the high-level architectural principles (redundancy, load distribution, risk management) with the specific technical advantages mentioned in the context (fault tolerance, scalability, parallel processing, node failure resilience)."}

{"step": "result", "content": "Distributed computing offers several key advantages over centralized systems by applying principles of redundancy and load distribution: 1) Fault tolerance - the system remains operational even when individual components fail, eliminating single points of failure, 2) Scalability - computational capacity can be expanded by adding more nodes rather than upgrading a single powerful machine, 3) Parallel processing - multiple tasks can be executed simultaneously across different nodes, significantly improving performance for complex workloads. Unlike centralized systems that create vulnerability through concentration, distributed architectures spread risk and resources across multiple nodes."}
"""

SYSTEM_PROMPT = f"""
You are an advanced AI assistant that combines Step-Back Abstraction with detailed Chain-of-Thought reasoning to provide comprehensive, well-reasoned responses based on the provided context.

Context: {{context}}

METHODOLOGY:
You must use both Step-Back Abstraction and Chain-of-Thought reasoning as demonstrated in these examples:

{ADVANCED_FEW_SHOT_EXAMPLES}

STEP-BY-STEP PROCESS:
1. **ANALYSE (Step-Back Abstraction)**: 
   - Identify the broader conceptual category of the question
   - Consider fundamental principles that apply across domains
   - Think about the high-level patterns and universal concepts

2. **THINK (Chain-of-Thought Reasoning)**:
   - Break down the specific query into components
   - Apply the abstract principles to the concrete context
   - Reason through the logical connections step by step
   - Connect general principles to specific details from the context
   - Use multiple "think" steps to build comprehensive understanding

3. **OUTPUT**: Synthesize both abstract understanding and detailed reasoning into a comprehensive response

4. **VALIDATE**: Check that your response captures both high-level concepts and specific contextual details

5. **RESULT**: Provide the final answer that demonstrates both conceptual depth and practical specificity

CRITICAL REQUIREMENTS:
- Always start with Step-Back Abstraction to identify broader principles
- Use multiple detailed Chain-of-Thought steps to reason through specifics
- Base all conclusions strictly on the provided context
- Connect abstract concepts to concrete details from the context
- If context is insufficient, clearly state this in your analysis
- Maintain JSON format: {{"step": "...", "content": "..."}}

RESPONSE QUALITY INDICATORS:
✓ Demonstrates understanding of broader conceptual frameworks
✓ Shows detailed logical reasoning through specific elements
✓ Connects abstract principles to contextual specifics
✓ Provides comprehensive, well-structured final answer
✓ Validates reasoning against both conceptual soundness and contextual accuracy
"""

# Main execution
query = input("> ")

print("\n" + "="*80)
print("🚀 ADVANCED RAG WITH HyDE + STEP-BACK + CHAIN-OF-THOUGHT")
print("="*80 + "\n")

# Step 1: HyDE-enhanced retrieval
print("📊 PHASE 1: HyDE-Enhanced Document Retrieval")
print("-" * 50)
relevant_chunks = hyde_retrieval(query, retriever, embedder, k=5)

print(f"\n📋 Retrieved {len(relevant_chunks)} most relevant chunks for reasoning")

# Step 2: Advanced reasoning with retrieved context
print("\n" + "="*50)
print("🧠 PHASE 2: Advanced Reasoning Process")
print("="*50 + "\n")

# Format context for the prompt
context_text = "\n\n".join([f"Chunk {i+1}: {chunk.page_content}" for i, chunk in enumerate(relevant_chunks)])

messages = [
    {"role": "system", "content": SYSTEM_PROMPT.format(context=context_text)},
    {"role": "user", "content": query},
]

print("🚀 Starting Advanced Reasoning Process...\n")

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=messages
    )
    
    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({"role": "assistant", "content": json.dumps(parsed_response)})
    
    # Enhanced output formatting
    step = parsed_response['step'].upper()
    content = parsed_response['content']
    
    if step == "ANALYSE":
        print(f"🔍 {step} (Step-Back Abstraction): {content}")
    elif step == "THINK":
        print(f"🧠 {step} (Chain-of-Thought): {content}")
    elif step == "OUTPUT":
        print(f"📋 {step}: {content}")
    elif step == "VALIDATE":
        print(f"✅ {step}: {content}")
    elif step == "RESULT":
        print(f"🎯 {step}: {content}")
    else:
        print(f"⚡ {step}: {content}")
    
    print()  # Add spacing between steps
    
    if parsed_response.get("step") == "result":
        break

print("✨ Advanced reasoning process completed!")
print("\n" + "="*80)
print("🏁 RAG PIPELINE FINISHED")
print("="*80)