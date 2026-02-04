import os
import httpx
import re
import json
import logging
from langgraph.graph import StateGraph, START, END

# ✅ IMPORT STATE FROM state.py
from .state import RLMState

# Import your tool
from .tools import recursive_document_search

# --- 1. CONFIGURATION ---
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RECURSION_DEPTH = 5
CONFIDENCE_THRESHOLD = 0.8
MODEL = "openai/gpt-oss-120b"  # Changed to a more reliable model

logger = logging.getLogger("ToolDebugger")
logger.setLevel(logging.INFO)

async def call_model(state: RLMState):
    current_depth = state.get("depth", 0)
    
    system_content = (
        f"Attempt {current_depth + 1}/{MAX_RECURSION_DEPTH}. "
        "Document loaded as `doc` (markdown).\n\n"
        "TOOL: recursive_document_search - assigns to `result`. "
        "Only: `import re` or `import json`.\n\n"
        "If tool returns no matches, refine your search pattern and try again.\n"
        "After getting useful tool results, respond with:\n"
        '{"answer": "...", "confidence": 0.0-1.0}\n'
        "No markdown blocks, just JSON."
    )

    # LIMIT HISTORY TO LAST 5 MESSAGES
    formatted_messages = [{"role": "system", "content": system_content}]
    recent_messages = state["messages"][-5:] if len(state["messages"]) > 5 else state["messages"]
    
    MAX_MESSAGE_LENGTH = 3000
    for msg in recent_messages:
        if isinstance(msg, dict):
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
        else:
            role = "user" if getattr(msg, "type", "") == "human" else "assistant"
            content = getattr(msg, "content", "")
        
        # Truncate long messages
        if len(content) > MAX_MESSAGE_LENGTH:
            content = content[:MAX_MESSAGE_LENGTH] + "\n[...truncated]"
        
        formatted_messages.append({"role": role, "content": content})

    logger.info(f"=== DEPTH {current_depth + 1} === Sending {len(formatted_messages)} messages")

    # PAYLOAD
    payload = {
        "model": MODEL,
        "messages": formatted_messages,
        "temperature": 0.7,
        "max_tokens": 1000,
        "tools": [{
            "type": "function",
            "function": {
                "name": "recursive_document_search",
                "description": "Run Python on doc. Assign to `result`.",
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
        logger.error(f"API Error: {response.status_code} - {response.text}")
        return {
            "messages": [{"role": "assistant", "content": f"API Error: {response.text}"}],
            "depth": current_depth + 1,
            "confidence": 0.0,
            "search_history": [f"Depth {current_depth+1}: API Error"]
        }

    data = response.json()
    message = data["choices"][0]["message"]
    tool_calls = message.get("tool_calls")
    
    new_messages = [message]
    history_entry = f"Depth {current_depth+1}: Processing"

    # Tool execution
    if tool_calls:
        logger.info(f"Tool calls detected: {len(tool_calls)}")
        for tool_call in tool_calls:
            if tool_call["function"]["name"] == "recursive_document_search":
                args = json.loads(tool_call["function"]["arguments"])
                code_to_run = args.get("code", "")
                
                logger.info(f"Executing tool with code length: {len(code_to_run)}")
                
                tool_output = recursive_document_search.invoke({"code": code_to_run})
                
                logger.info(f"Tool output length: {len(str(tool_output))}")
                
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": "recursive_document_search",
                    "content": str(tool_output)
                }
                new_messages.append(tool_message)
                history_entry = f"Depth {current_depth+1}: Tool executed (Output len: {len(str(tool_output))})"

    logger.info(f"Adding {len(new_messages)} new messages to state")
    if new_messages:
        last_new_msg = new_messages[-1]
        logger.info(f"Last new message role: {last_new_msg.get('role', 'N/A')}")

    # Parse confidence
    content = message.get("content", "") or ""
    logger.info(f"Raw LLM content length: {len(content)}")
    
    confidence = 0.0
    
    try:
        cleaned_content = re.sub(r'^```json\s*|\s*```$', '', content.strip(), flags=re.MULTILINE)
        json_data = json.loads(cleaned_content)
        confidence = float(json_data.get("confidence", 0.0))
        logger.info(f"Parsed confidence from JSON: {confidence}")
    except:
        conf_match = re.search(r'"confidence"\s*:\s*(1(?:\.0+)?|0(?:\.\d+)?)', content)
        if conf_match:
            confidence = float(conf_match.group(1))
            logger.info(f"Parsed confidence from regex: {confidence}")
        else:
            logger.warning(f"Could not parse confidence")

    return {
        "messages": state["messages"] + new_messages,
        "depth": current_depth + 1,
        "confidence": confidence,
        "search_history": [history_entry]
    }

def should_continue(state: RLMState):
    logger.info(f"=== ROUTING === Depth: {state['depth']}, Confidence: {state['confidence']}")
    logger.info(f"Total messages in state: {len(state['messages'])}")
    
    if state["depth"] >= MAX_RECURSION_DEPTH:
        logger.info("Exiting: Max recursion depth reached")
        return "exit"
        
    if state["confidence"] >= CONFIDENCE_THRESHOLD:
        logger.info("Exiting: Confidence threshold met")
        return "exit"
    
    # Check if we have messages
    if not state["messages"]:
        logger.warning("No messages in state!")
        return "exit"
    
    last_msg = state["messages"][-1]
    
    # Debug: print the last message structure
    logger.info(f"Last message type: {type(last_msg)}")
    
    # Handle both dict and object types
    if isinstance(last_msg, dict):
        last_role = last_msg.get("role", "unknown")
    else:
        last_role = getattr(last_msg, "role", getattr(last_msg, "type", "unknown"))
    
    logger.info(f"Last message role: {last_role}")
    
    # If it's a tool message, continue to let agent synthesize
    if last_role == "tool":
        logger.info("Continuing to agent: Tool result needs processing")
        return "agent"
    
    logger.info("Exiting: Agent has provided final response")
    return "exit"

# --- 4. GRAPH CONSTRUCTION ---
builder = StateGraph(RLMState)

builder.add_node("agent", call_model)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "agent": "agent",
        "exit": END
    }
)

graph = builder.compile()