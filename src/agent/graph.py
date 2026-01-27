import os
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from .tools import recursive_document_search

# A simple reducer to append new history items
def update_history(left: List[str], right: List[str]) -> List[str]:
    return left + right

class RLMState(TypedDict):
    messages: Annotated[list, add_messages]
    # Tracks the code/offsets tried: e.g., ["Offset 500: ToC", "Offset 1200: Sidebar"]
    search_history: Annotated[List[str], update_history]

# Updated for GPT-OSS 120B specific parameters
llm = ChatOpenAI(
    model="openai/gpt-oss-120b", 
    openai_api_key=os.getenv("OPENROUTER_API_KEY"), # Use OpenRouter Key
    base_url="https://openrouter.ai/api/v1",
    temperature=1.0, # GPT-OSS performs better at temp 1.0 with reasoning enabled
    model_kwargs={
        "extra_body": {
            "reasoning_effort": "high" # Ensures deep logic for regex/slicing
        }
    },
    default_headers={
        "HTTP-Referer": "http://localhost:2024", 
        "X-Title": "LangGraph Agent",
    }
)

# Tool binding remains the same
tools = [recursive_document_search]
llm_with_tools = llm.bind_tools(tools)

def call_model(state: RLMState):
    # GPT-OSS uses a special 'Harmony' format. 
    # System messages are highly effective here.
    system_msg = {
        "role": "system", 
        "content": (
        "You are a precise document extractor. When using 'recursive_document_search':\n"
        "1. Your 'code' argument must be a clean string of Python code.\n"
        "2. Do NOT wrap the code in markdown (no ```python blocks).\n"
        "3. Do NOT use the 'import' keyword; the 're' module is already available as 're'.\n"
        "4. Start your code directly with the logic (e.g., match = re.search...)."
        )
    }
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Graph construction remains correct
builder = StateGraph(RLMState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()