import re
import json
import logging
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import guarded_iter_unpack_sequence, full_write_guard
import pymupdf4llm
from langchain_core.tools import tool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ToolDebugger")

# Pre-load document content
try:
    # This is the only place the document source is defined
    MANUAL_CONTENT = pymupdf4llm.to_markdown("data/six-annual-report-2024-en.pdf")
except Exception as e:
    MANUAL_CONTENT = "Error loading PDF: " + str(e)

@tool
def recursive_document_search(code: str):
    """
    Safely execute dynamic Python code in a sandbox to extract data from 'doc'.
    The LLM does NOT need to import 're' or 'json' as they are pre-injected.
    """
    # 1. Clean AI markdown formatting
    clean_code = re.sub(r'^```python\n|```$', '', code, flags=re.MULTILINE).strip()

    # 2. SANITIZATION: Strip 'import' and 'from' keywords.
    clean_code = re.sub(r'^(import|from)\s+.*$', '', clean_code, flags=re.MULTILINE).strip()

    logger.info("Executing Sanitized Sandbox Code:\n%s", clean_code)

    # 3. Create the Safe Environment
    safe_env = safe_globals.copy()
    safe_env.update({
        # Data
        "doc": MANUAL_CONTENT,
        
        # Pre-injected modules
        "re": re,
        "json": json,
        
        # Internal state
        "result": None,

        # Security Guards for RestrictedPython
        "_getattr_": getattr,
        "_write_": full_write_guard,
        "_getitem_": default_guarded_getitem,
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_print_": PrintCollector,

        # Built-ins required for basic logic
        "len": len, 
        "str": str, 
        "int": int, 
        "float": float,
        "list": list, 
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "range": range,
        "bool": bool,
        "enumerate": enumerate,  # ← ADD THIS
        "zip": zip,              # ← ADD THIS
        "map": map,              # ← ADD THIS
        "filter": filter,        # ← ADD THIS
        "sorted": sorted,        # ← ADD THIS
        "min": min,              # ← ADD THIS
        "max": max,              # ← ADD THIS
        "sum": sum,              # ← ADD THIS
        "abs": abs,              # ← ADD THIS
        "round": round,          # ← ADD THIS
        "any": any,              # ← ADD THIS
        "all": all,              # ← ADD THIS
    })

    try:
        byte_code = compile_restricted(clean_code, filename='<inline>', mode='exec')
        exec(byte_code, safe_env)
        
        final_result = safe_env.get('result')
        
        if final_result is None:
            return "Sandbox executed, but the variable 'result' was not assigned any value."
        
        result_str = str(final_result)
        
        # Return message about empty results for LLM to handle
        if len(result_str.strip()) == 0:
            return "Search completed but found no matches. The pattern did not match any content in the document."
        
        MAX_RESULT_LENGTH = 2000
        
        if len(result_str) > MAX_RESULT_LENGTH:
            logger.warning(f"Tool result truncated from {len(result_str)} to {MAX_RESULT_LENGTH} chars")
            return result_str[:MAX_RESULT_LENGTH] + "\n\n[...truncated]"
            
        return result_str

    except Exception as e:
        return f"Sandbox Security/Execution Error: {str(e)}"