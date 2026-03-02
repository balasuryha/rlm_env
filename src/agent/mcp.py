import asyncio
import os
import httpx
from typing import List, TypedDict, Annotated, Dict
from operator import add

# LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ChromaDB
import chromadb
from chromadb.utils import embedding_functions

# MCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- 1. CONFIGURATION & INSECURE CLIENTS ---
BASE_URL = "YOUR_CUSTOM_ENDPOINT_URL"
API_KEY = "YOUR_API_KEY"
EMBED_MODEL = "your-embedding-model"
PDF_DIR = os.path.abspath("./my_100_pdfs")

# Global insecure clients
insecure_async_client = httpx.AsyncClient(verify=False)
insecure_sync_client = httpx.Client(verify=False)

# Configure the LLM (Orchestrator & Workers)
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model="your-llm-model",
    temperature=0.1,
    http_client=insecure_sync_client,
    http_async_client=insecure_async_client
)

# --- 2. STATE DEFINITION ---
class SwarmState(TypedDict):
    query: str
    vector_results: List[Dict]
    grouped_docs: Dict[str, List[int]]
    worker_reports: Annotated[List[str], add] # Parallel results accumulate here
    final_answer: str

# --- 3. GRAPH NODES ---

def retrieve_node(state: SwarmState):
    """Step 1: Use ChromaDB with an insecure embedding client."""
    client = chromadb.PersistentClient(path="./chroma_db_DFS")
    
    # Custom Embedding Function with verify=False
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=API_KEY,
        api_base=BASE_URL,
        model_name=EMBED_MODEL
    )
    # Patch the internal Chroma/OpenAI client to skip SSL
    ef._client.http_client = insecure_sync_client
    
    collection = client.get_collection(name="pdf_embeddings", embedding_function=ef)
    results = collection.query(query_texts=[state["query"]], n_results=30)
    
    chunks = []
    for i in range(len(results['ids'][0])):
        chunks.append({
            "metadata": results['metadatas'][0][i],
            "text": results['documents'][0][i]
        })
    return {"vector_results": chunks}

def group_node(state: SwarmState):
    """Step 2: Group 30 chunks into Top 5 unique files."""
    groups = {}
    for chunk in state["vector_results"]:
        fname = chunk['metadata'].get('fileName') or chunk['metadata'].get('filename')
        page = int(chunk['metadata'].get('page_number', 1))
        groups.setdefault(fname, set()).add(page)
    
    # Slice to top 5 and sort pages
    top_5 = {k: sorted(list(v)) for k, v in list(groups.items())[:5]}
    return {"grouped_docs": top_5}

async def worker_swarm_node(state: SwarmState):
    """Step 3: Parallel Fan-out to MCP + Sub-LLMs."""
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@sylphx/pdf-reader-mcp"],
        cwd=PDF_DIR
    )

    async def run_single_worker(filename, pages):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Fetch full context from PDF via MCP
                p_str = ",".join(map(str, pages))
                mcp_res = await session.call_tool("read_pdf", arguments={
                    "sources": [{"path": filename, "pages": p_str}],
                    "include_full_text": True
                })
                pdf_text = mcp_res.content[0].text
                
                # Analyze with LLM (Worker)
                prompt = f"Doc: {filename}\nPages: {p_str}\nContent: {pdf_text}\nQuery: {state['query']}"
                res = await llm.ainvoke([
                    ("system", "Extract relevant facts and cite the page numbers."),
                    ("user", prompt)
                ])
                return f"--- Report: {filename} ---\n{res.content}"

    # Execute all 5 workers in parallel
    tasks = [run_single_worker(f, p) for f, p in state["grouped_docs"].items()]
    reports = await asyncio.gather(*tasks)
    return {"worker_reports": reports}

async def synthesis_node(state: SwarmState):
    """Step 4: Fan-in to the Final Orchestrator Answer."""
    combined = "\n\n".join(state["worker_reports"])
    final = await llm.ainvoke([
        ("system", "Synthesize the provided document reports into one comprehensive final answer."),
        ("user", f"User Question: {state['query']}\n\nReports:\n{combined}")
    ])
    return {"final_answer": final.content}

# --- 4. COMPILE GRAPH ---
builder = StateGraph(SwarmState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("group", group_node)
builder.add_node("swarm", worker_swarm_node)
builder.add_node("synthesize", synthesis_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "group")
builder.add_edge("group", "swarm")
builder.add_edge("swarm", "synthesize")
builder.add_edge("synthesize", END)

app = builder.compile()

# --- 5. EXECUTION ---
async def run_system(user_query: str):
    print(f"🚀 Starting Swarm for: {user_query}")
    try:
        inputs = {"query": user_query, "worker_reports": []}
        final_state = await app.ainvoke(inputs)
        print("\n" + "="*50 + "\nFINAL ANSWER:\n" + "="*50)
        print(final_state["final_answer"])
    finally:
        # Cleanup HTTP connections
        await insecure_async_client.aclose()
        insecure_sync_client.close()

if __name__ == "__main__":
    asyncio.run(run_system("What are the chemical handling protocols?"))