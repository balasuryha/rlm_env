import os
import httpx
import re
import json
import logging
from langgraph.graph import StateGraph, START, END
from .state import RLMState
from .tools import recursive_document_search

# --- CONFIG ---
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RECURSION_DEPTH = 3
CONFIDENCE_THRESHOLD = 0.8
MODEL = "openai/gpt-oss-120b"
TIMEOUT = 60.0

logger = logging.getLogger("ToolDebugger")

def serialize_message(msg):
    if isinstance(msg, dict):
        return {"role": msg.get("role", "assistant"), "content": str(msg.get("content", ""))}
    if hasattr(msg, "content"):
        role = "user" if getattr(msg, "type", "") == "human" else "assistant"
        return {"role": role, "content": str(msg.content)}
    return {"role": "assistant", "content": str(msg)}

# --- AGENT NODE ---

async def call_model(state: RLMState):
    current_depth = state.get("depth", 0)
    user_query = state["messages"][0].content if state["messages"] else "Extract requested information."
    
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    # 1️⃣ PHASE ONE: ADAPTIVE CODE GENERATION
    # No hardcoding. The model is told to be "greedy" to capture full context.
    code_prompt = f"""You are a Python extraction agent. (Attempt {current_depth + 1})
Document string: `doc`. Modules: `re`, `json`.

TASK: Satisfy the user request: "{user_query}"

STRATEGY:
1. Use `re.finditer` or `re.search` to find the most relevant header or keywords.
2. Since PDFs converted to Markdown can have messy headers (e.g., "# 1.2 Risk Management"), use flexible regex like `r'#+.*Risk.*Management.*'`.
3. Capture a large block of text (e.g., 2000-4000 characters) following the match to ensure you don't cut off the content.
4. If a specific section is not found, return the first 20 lines of the document so we can see the format.

RULES:
- NO 'import' statements.
- The modules re and json are already available. Do not use the import keyword.
- Assign the final text to the variable `result`.
- Return ONLY valid JSON: {{"python_code": "..."}}
"""

    msgs = [{"role": "system", "content": code_prompt}] + [serialize_message(m) for m in state["messages"]]

    async with httpx.AsyncClient(verify=False, timeout=TIMEOUT) as client:
        try:
            # 1. Generate Python snippet
            res_code = await client.post(OPENROUTER_URL, headers=headers, json={
                "model": MODEL, "messages": msgs, "temperature": 0.0
            })
            res_code.raise_for_status()
            
            raw_code_resp = res_code.json()["choices"][0]["message"]["content"]
            json_match = re.search(r'\{.*\}', raw_code_resp, re.S).group()
            code = json.loads(json_match)["python_code"]
            
            # 2. Run in Tool
            tool_output = recursive_document_search.invoke({"code": code})
            logger.info(f"Tool Output Sample: {str(tool_output)[:100]}...")
            
        except Exception as e:
            tool_output = f"Critical extraction error: {str(e)}"

        # 2️⃣ PHASE TWO: INTELLIGENT SYNTHESIS
        synth_prompt = f"""
        Original User Query: {user_query}
        Data Extracted from Doc: {tool_output}
        
        Instruction: 
        Analyze the data. If it looks like a list of headers or random text, state that you couldn't find the specific answer.
        Otherwise, answer the query based ONLY on the extracted text.
        Return JSON: {{"answer": "...", "confidence": 0.0-1.0}}
        """
        
        res_synth = await client.post(OPENROUTER_URL, headers=headers, json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Return ONLY JSON with 'answer' and 'confidence'."},
                {"role": "user", "content": synth_prompt}
            ],
            "temperature": 0.0
        })
        
        try:
            synth_json = re.search(r'\{.*\}', res_synth.json()["choices"][0]["message"]["content"], re.S).group()
            final_data = json.loads(synth_json)
            answer = final_data.get("answer", "Parsing error in synthesis.")
            conf = float(final_data.get("confidence", 0.0))
            final_message = f"{answer}\n\n---\n**Confidence Score:** {int(conf * 100)}%"
        except:
            answer = "The system failed to interpret the document results."
            conf = 0.0

    # 3️⃣ UPDATE STATE
    # Note: Jan 27 Update - appending to search_history list.
    return {
        "messages": [{"role": "assistant", "content": answer}],
        "depth": current_depth + 1,
        "confidence": conf,
        "search_history": [f"Depth {current_depth+1}: Query '{user_query[:15]}', Output Len {len(str(tool_output))}, , Confidence: {conf:.2f}"]
    }

# --- ROUTER ---

def route_decision(state: RLMState):
    if state["confidence"] >= CONFIDENCE_THRESHOLD or state["depth"] >= MAX_RECURSION_DEPTH:
        return "end"
    return "continue"

# --- GRAPH ---

builder = StateGraph(RLMState)
builder.add_node("agent", call_model)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_decision, {"continue": "agent", "end": END})

graph = builder.compile()