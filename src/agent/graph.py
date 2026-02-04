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
MODEL = "openai/gpt-oss-120b" 

logger = logging.getLogger("ToolDebugger")
logger.setLevel(logging.INFO)

async def call_model(state: RLMState):
    current_depth = state.get("depth", 0)
    
    # Updated System Prompt to request Markdown instead of Tool Calls
    system_content = (
        f"Attempt {current_depth + 1}/{MAX_RECURSION_DEPTH}. "
        "Document loaded as `doc` (markdown).\n\n"
        "To search the document, write a Python code block using `re` or `json`. "
        "Assign your findings to the variable `result`.\n\n"
        "FORMAT REQUIRED:\n"
        "```python\n"
        "import re\n"
        "result = re.findall('pattern', doc)\n"
        "```\n\n"
        "After getting results, respond with your final answer and confidence in JSON:\n"
        '{"answer": "...", "confidence": 0.0-1.0}'
    )

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
        
        if len(content) > MAX_MESSAGE_LENGTH:
            content = content[:MAX_MESSAGE_LENGTH] + "\n[...truncated]"
        
        formatted_messages.append({"role": role, "content": content})

    logger.info(f"=== DEPTH {current_depth + 1} === Sending {len(formatted_messages)} messages")

    # REMOVED official "tools" parameter from payload
    payload = {
        "model": MODEL,
        "messages": formatted_messages,
        "temperature": 0.4, # Lowered slightly for more stable code generation
        "max_tokens": 1000
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
    content = message.get("content", "") or ""
    
    new_messages = [message]
    history_entry = f"Depth {current_depth+1}: Processing"

    # --- MANUAL CODE EXTRACTION (REGEX) ---
    code_match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
    
    if code_match:
        code_to_run = code_match.group(1)
        logger.info(f"Extracted code block length: {len(code_to_run)}")
        
        # Execute tool locally
        tool_output = recursive_document_search.invoke({"code": code_to_run})
        
        # Create a feedback message from 'user' perspective so the LLM sees the result
        tool_message = {
            "role": "user",
            "content": f"CODE_EXECUTION_RESULT:\n{tool_output}"
        }
        new_messages.append(tool_message)
        history_entry = f"Depth {current_depth+1}: Local code executed"

    # Parse confidence
    confidence = 0.0
    try:
        cleaned_content = re.sub(r'^```json\s*|\s*```$', '', content.strip(), flags=re.MULTILINE)
        json_data = json.loads(cleaned_content)
        confidence = float(json_data.get("confidence", 0.0))
    except:
        conf_match = re.search(r'"confidence"\s*:\s*(1(?:\.0+)?|0(?:\.\d+)?)', content)
        if conf_match:
            confidence = float(conf_match.group(1))

    return {
        "messages": state["messages"] + new_messages,
        "depth": current_depth + 1,
        "confidence": confidence,
        "search_history": [history_entry]
    }

def should_continue(state: RLMState):
    logger.info(f"=== ROUTING === Depth: {state['depth']}, Confidence: {state['confidence']}")
    
    # 1. Check for Max Depth first to prevent infinite loops
    if state["depth"] >= MAX_RECURSION_DEPTH:
        logger.info("Exiting: Max recursion depth reached.")
        return "exit"
    
    # 2. Get the content of the last message
    if not state["messages"]:
        return "exit"
        
    last_msg = state["messages"][-1]
    if isinstance(last_msg, dict):
        last_content = last_msg.get("content", "")
    else:
        last_content = getattr(last_msg, "content", "")

    # 3. CRITICAL CHECK: If the code execution was empty, force a retry.
    # We ignore confidence here because the LLM often "hallucinates" confidence 
    # before seeing that its regex failed.
    if "CODE_EXECUTION_RESULT: []" in last_content or "found no matches" in last_content:
        logger.info("Search returned no data. Forcing agent to try a different pattern.")
        return "agent"

    # 4. If we have actual data or a final answer, check confidence
    if state["confidence"] >= CONFIDENCE_THRESHOLD:
        logger.info("Exiting: Confidence threshold met.")
        return "exit"
    
    # 5. If we have a code result that isn't empty, but confidence is still low, 
    # the agent needs to synthesize the answer.
    if "CODE_EXECUTION_RESULT" in last_content:
        logger.info("Continuing: Agent needs to analyze the search results.")
        return "agent"

    # Default fallback: keep going until thresholds are met
    return "agent"

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