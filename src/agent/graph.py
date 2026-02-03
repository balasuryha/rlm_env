import os
import httpx
import re
from typing import List
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

# ✅ IMPORT STATE FROM state.py
from .state import RLMState

# Import your tool
from .tools import recursive_document_search

# --- 1. CONFIGURATION ---
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RECURSION_DEPTH = 5
CONFIDENCE_THRESHOLD = 0.8

# --- 2. THE AGENT NODE ---
async def call_model(state: RLMState):
    current_depth = state.get("depth", 0)

    system_content = (
    f"You are on attempt {current_depth + 1} of {MAX_RECURSION_DEPTH}.\n"
    "A document is ALREADY LOADED as a STRING (markdown text) in `doc`.\n"
    "Use ONLY regex or string operations to search it.\n\n"
    "TOOL RULES (MANDATORY):\n"
    "1. Tool code MUST assign its output to a variable named `result`.\n"
    "2. Tool code MUST NOT return JSON.\n"
    "3. Tool code MUST NOT print.\n"
    "4. The agent (not the tool) produces the final JSON answer.\n\n"
    "IMPORT RULES (STRICT):\n"
    "5. You MAY ONLY use: `import re` OR `import json`\n"
    "6. You MUST NOT import any other module.\n"
    "7. Do NOT use textwrap, sys, math, pathlib, or any other libraries.\n"
    "8. If you import anything else, execution will fail.\n\n"
    "EXECUTION RULES:\n"
    "9. You MUST call the tool at least once before answering.\n"
    "10. If you answer without calling the tool, the response is invalid.\n\n"
    "Final agent response format:\n"
    "{ \"answer\": \"...\", \"confidence\": 0.0-1.0 }\n"
)


    # --- STRICT ROLE MAPPING ---
    formatted_messages = [{"role": "system", "content": system_content}]

    for msg in state["messages"]:
        if isinstance(msg, dict):
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", "assistant")
            content = getattr(msg, "content", "")

        role = "user" if role in ("human", "user") else "assistant"
        formatted_messages.append({"role": role, "content": content})

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": formatted_messages,
        "temperature": 1.0,
        "extra_body": {"reasoning_effort": "high"},
        "tools": [{
            "type": "function",
            "function": {
                "name": "recursive_document_search",
                "description": "Run restricted Python on the document. Assign output to `result`.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"}
                    },
                    "required": ["code"]
                }
            }
        }]
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "HTTP-Referer": "http://localhost:2024",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(verify=False) as client:
        response = await client.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=60.0
        )

    if response.status_code != 200:
        return {
            "messages": [{"role": "assistant", "content": response.text}],
            "search_history": [f"Depth {current_depth}: API error"],
            "confidence": 0.0,
            "depth": current_depth + 1
        }

    data = response.json()
    message = data["choices"][0]["message"]
    content = message.get("content", "") or ""

    confidence = 0.0
    match = re.search(r'"confidence"\s*:\s*(1(?:\.0+)?|0(?:\.\d+)?)', content)
    if match:
        confidence = float(match.group(1))

    history = (
        ["Tool invoked"]
        if message.get("tool_calls")
        else [f"Answered with confidence {confidence}"]
    )

    return {
        "messages": state["messages"] + [message],
        "depth": current_depth + 1,
        "confidence": confidence,
        "search_history": history
    }


# --- 3. ROUTING LOGIC ---
def should_continue(state: RLMState):
    # Always allow tool on first pass
    if state["depth"] == 1:
        return "tools"

    if state["depth"] >= MAX_RECURSION_DEPTH:
        return "exit"

    if state["confidence"] >= CONFIDENCE_THRESHOLD:
        return "exit"

    last = state["messages"][-1]
    tool_calls = last.get("tool_calls") if isinstance(last, dict) else None

    return "tools" if tool_calls else "exit"



# --- 4. GRAPH CONSTRUCTION ---
builder = StateGraph(RLMState)

builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode([recursive_document_search]))

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "exit": END
    }
)

builder.add_edge("tools", "agent")


graph = builder.compile()
