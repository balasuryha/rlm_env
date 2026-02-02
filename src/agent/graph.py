import os
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
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
    current_depth = state.get("depth", 0)
    
    system_msg = {
        "role": "system", 
        "content": (
            f"You are on attempt {current_depth + 1} of {MAX_RECURSION_DEPTH}.\n"
            "After extracting data, evaluate your confidence (0.0-1.0).\n"
            "If the data is missing or ambiguous, keep confidence LOW.\n"
            "You must return your response in JSON format if not calling a tool:\n"
            "{'answer': '...', 'confidence': 0.9}"
            "You are a document extractor. IMPORTANT: The document contains a Table of Contents (ToC).\n"
            "If your search returns a list of page numbers, you are in the ToC. \n"
            "You must look deeper in the document (higher character offsets) to find the actual content.\n"
            "Always assign your final string to the variable 'result'."
        )
    }
    
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    # Logic to parse confidence from the AI's prose if it's not a tool call
    conf_value = 0.0
    if not response.tool_calls:
        # Simple extraction logic (or use structured output)
        if "confidence" in response.content:
            # (Logic to extract float from text)
            conf_value = 0.85 

    return {
        "messages": [response],
        "depth": current_depth + 1,
        "confidence": conf_value
    }

MAX_RECURSION_DEPTH = 5
CONFIDENCE_THRESHOLD = 0.8

def should_continue(state: RLMState):
    # 1. Check if we've hit the recursion limit (Safety Valve)
    if state.get("depth", 0) >= MAX_RECURSION_DEPTH:
        return "exit"
    
    # 2. Check if the AI has expressed high confidence in its current answer
    if state.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD:
        return "exit"
    
    # 3. Inspect the last message
    last_message = state["messages"][-1]
    
    # 4. If the AI wants to use the search tool, go to 'tools'
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # 5. If no tool calls and no high confidence, but it provided an answer, 
    # we exit (or you could force a retry if you want to be stricter).
    return "exit"


# Graph construction remains correct
builder = StateGraph(RLMState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
builder.add_conditional_edges(
    "agent", 
    should_continue, 
    {
        "tools": "tools", 
        "exit": END
    }
)

graph = builder.compile()