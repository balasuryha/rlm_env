import asyncio
import os
from typing import List, TypedDict, Annotated, Dict
from operator import add

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import chromadb
from chromadb.utils import embedding_functions

# --- 1. CONFIGURATION ---
# Replace with your custom credentials from your indexing setup
CUSTOM_BASE_URL = "https://your-custom-endpoint.com/v1"
API_KEY = "your-api-key"
EMBED_MODEL = "text-embedding-3-small" # Your chosen model
PDF_DIR = os.path.abspath("./my_100_pdfs")

# Custom LLM Setup
llm = ChatOpenAI(
    base_url=CUSTOM_BASE_URL,
    api_key=API_KEY,
    model="your-model-name", 
    temperature=0
)

# --- 2. STATE DEFINITION ---
class SwarmState(TypedDict):
    query: str
    vector_results: List[Dict]      # Chunks from Chroma
    grouped_docs: Dict[str, List]   # {filename: [page_numbers]}
    worker_reports: Annotated[List[str], add] # "add" merges parallel results
    final_answer: str

# --- 3. NODES ---

def retrieve_from_chroma(state: SwarmState):
    """Step 1: Retrieve 30 chunks using your existing ChromaDB setup."""
    client = chromadb.PersistentClient(path="./chroma_db_DFS")
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=API_KEY,
        api_base=CUSTOM_BASE_URL,
        model_name=EMBED_MODEL
    )
    collection = client.get_collection(name="pdf_embeddings", embedding_function=embedding_fn)
    
    # Query for the top 30 chunks
    results = collection.query(query_texts=[state["query"]], n_results=30)
    
    chunks = []
    for i in range(len(results['ids'][0])):
        chunks.append({
            "metadata": results['metadatas'][0][i],
            "text": results['documents'][0][i]
        })
    return {"vector_results": chunks}

def group_documents(state: SwarmState):
    """Step 2: Group by filename and pick top 5 unique documents."""
    groups = {}
    for chunk in state["vector_results"]:
        fname = chunk['metadata']['fileName'] # Use your metadata key
        page = int(chunk['metadata'].get('page_number', 1))
        groups.setdefault(fname, set()).add(page)
    
    top_5 = {k: sorted(list(v)) for k, v in list(groups.items())[:5]}
    return {"grouped_docs": top_5}

async def swarm_worker_node(state: SwarmState):
    """Step 3: Parallel MCP retrieval and Worker analysis."""
    # Define the MCP server parameters (Sylphx uses Node.js)
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@sylphx/pdf-reader-mcp"],
        cwd=PDF_DIR  # Resolves filenames relative to this directory
    )

    async def run_worker(filename, pages):
        """Single worker task: MCP Fetch -> LLM Analysis."""
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Fetch full context via MCP
                pages_str = ",".join(map(str, pages))
                mcp_res = await session.call_tool("read_pdf", arguments={
                    "sources": [{"path": filename, "pages": pages_str}],
                    "include_full_text": True
                })
                full_text = mcp_res.content[0].text
                
                # Custom LLM call for this document
                worker_prompt = f"Document: {filename}\nContent: {full_text}\nQuestion: {state['query']}"
                res = await llm.ainvoke([("system", "Extract relevant facts."), ("user", worker_prompt)])
                return f"Source {filename}: {res.content}"

    # FAN-OUT: Run all 5 workers in parallel
    tasks = [run_worker(fname, pgs) for fname, pgs in state["grouped_docs"].items()]
    reports = await asyncio.gather(*tasks)
    return {"worker_reports": reports}

async def final_synthesize(state: SwarmState):
    """Step 4: FAN-IN: Combine all reports into one answer."""
    combined = "\n\n".join(state["worker_reports"])
    summary = await llm.ainvoke([
        ("system", "Synthesize these document reports into a final cohesive answer."),
        ("user", f"User Question: {state['query']}\n\nReports:\n{combined}")
    ])
    return {"final_answer": summary.content}

# --- 4. GRAPH CONSTRUCTION ---
workflow = StateGraph(SwarmState)

workflow.add_node("retrieve", retrieve_from_chroma)
workflow.add_node("group", group_documents)
workflow.add_node("mcp_workers", swarm_worker_node)
workflow.add_node("synthesize", final_synthesize)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "group")
workflow.add_edge("group", "mcp_workers")
workflow.add_edge("mcp_workers", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()