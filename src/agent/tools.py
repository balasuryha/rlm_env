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
    # This prevents RestrictedPython from throwing "__import__ not found" errors.
    # Since we inject 're' and 'json' below, the code will still work.
    clean_code = re.sub(r'^(import|from)\s+.*$', '', clean_code, flags=re.MULTILINE).strip()

    logger.info("Executing Sanitized Sandbox Code:\n%s", clean_code)

    # 3. Create the Safe Environment
    safe_env = safe_globals.copy()
    safe_env.update({
        # Data
        "doc": MANUAL_CONTENT,
        
        # Pre-injected modules (LLM can use 're.search' directly)
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
        "list": list, 
        "dict": dict,
        "range": range,
        "bool": bool
    })

    try:
        # 4. Compile and Execute
        byte_code = compile_restricted(clean_code, filename='<inline>', mode='exec')
        exec(byte_code, safe_env)
        
        # 5. Extract the result
        final_result = safe_env.get('result')
        
        if final_result is None:
            return "Sandbox executed, but the variable 'result' was not assigned any value."
            
        return str(final_result)

    except Exception as e:
        # If the LLM still tried to do something forbidden, we catch it here
        return f"Sandbox Security/Execution Error: {str(e)}"