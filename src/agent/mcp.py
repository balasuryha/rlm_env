import asyncio
import os
import httpx
import argparse
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

# --- 1. CONFIGURATION ---
BASE_URL = "YOUR_CUSTOM_ENDPOINT_URL"
API_KEY = "YOUR_API_KEY"
EMBED_MODEL = "your-embedding-model"
PDF_DIR = os.path.abspath("./my_100_pdfs")

# SSL Bypass Clients
insecure_async_client = httpx.AsyncClient(verify=False)
insecure_sync_client = httpx.Client(verify=False)

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
    worker_reports: Annotated[List[str], add]
    final_answer: str

# --- 3. GRAPH NODES ---

def retrieve_node(state: SwarmState):
    client = chromadb.PersistentClient(path="./chroma_db_DFS")
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=API_KEY, api_base=BASE_URL, model_name=EMBED_MODEL
    )
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
    groups = {}
    for chunk in state["vector_results"]:
        fname = chunk['metadata'].get('fileName') or chunk['metadata'].get('filename')
        page = int(chunk['metadata'].get('page_number', 1))
        groups.setdefault(fname, set()).add(page)
    
    top_5 = {k: sorted(list(v)) for k, v in list(groups.items())[:5]}
    return {"grouped_docs": top_5}

async def swarm_worker_node(state: SwarmState):
    server_params = StdioServerParameters(
        command="npx", args=["-y", "@sylphx/pdf-reader-mcp"], cwd=PDF_DIR
    )

    async def run_single_worker(filename, pages):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                p_str = ",".join(map(str, pages))
                mcp_res = await session.call_tool("read_pdf", arguments={
                    "sources": [{"path": filename, "pages": p_str}],
                    "include_full_text": True
                })
                pdf_text = mcp_res.content[0].text
                
                res = await llm.ainvoke([
                    ("system", "Extract facts from the text and cite page numbers."),
                    ("user", f"Doc: {filename}\nContent: {pdf_text}\nQuery: {state['query']}")
                ])
                return f"--- Report: {filename} ---\n{res.content}"

    tasks = [run_single_worker(f, p) for f, p in state["grouped_docs"].items()]
    reports = await asyncio.gather(*tasks)
    return {"worker_reports": reports}

async def synthesis_node(state: SwarmState):
    combined = "\n\n".join(state["worker_reports"])
    final = await llm.ainvoke([
        ("system", "Synthesize the provided reports into a final answer."),
        ("user", f"Question: {state['query']}\n\nReports:\n{combined}")
    ])
    return {"final_answer": final.content}

# --- 4. COMPILE GRAPH ---
builder = StateGraph(SwarmState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("group", group_node)
builder.add_node("swarm", swarm_worker_node)
builder.add_node("synthesize", synthesis_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "group")
builder.add_edge("group", "swarm")
builder.add_edge("swarm", "synthesize")
builder.add_edge("synthesize", END)

app = builder.compile()

# --- 5. CMD EXECUTION LOGIC ---
async def run_cli():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="PDF Swarm Orchestrator CLI")
    parser.add_argument("question", nargs="?", help="The question you want to ask your PDFs")
    args = parser.parse_args()

    # If no question provided in CMD, ask for it interactively
    user_query = args.question
    if not user_query:
        user_query = input("❓ Enter your question: ")

    if not user_query.strip():
        print("❌ No question provided. Exiting.")
        return

    print(f"\n🔍 Processing: {user_query}...")
    
    try:
        final_state = await app.ainvoke({"query": user_query, "worker_reports": []})
        print("\n" + "="*60)
        print("🎯 FINAL ANSWER")
        print("="*60)
        print(final_state["final_answer"])
        print("="*60 + "\n")
    finally:
        await insecure_async_client.aclose()
        insecure_sync_client.close()

if __name__ == "__main__":
    asyncio.run(run_cli())