import re
import logging # Added for cleaner terminal output
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import guarded_iter_unpack_sequence
import pymupdf4llm
from langchain_core.tools import tool

# Setup logging to see the AI's code in your console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ToolDebugger")

# Pre-load content
MANUAL_CONTENT = pymupdf4llm.to_markdown("data/six-annual-report-2024-en.pdf")

def _safe_import(name, *args, **kwargs):
    if name == 're': return re
    raise ImportError(f"Import of {name} is forbidden.")

@tool
def recursive_document_search(code: str):
    """Safely execute Python code on MANUAL_CONTENT. 're' is pre-loaded."""
    
    # 1. CLEANING: Remove AI artifacts like ```python or leading spaces
    clean_code = code.replace("```python", "").replace("```", "").strip()
    
    # 2. LOGGING: Print the code so you can see it in your terminal
    print("\n" + "="*30)
    print("AI EXECUTING CODE:")
    print(clean_code)
    print("="*30 + "\n")

    safe_env = safe_globals.copy()
    safe_env.update({
        "MANUAL_CONTENT": MANUAL_CONTENT,
        "re": re,
        "__import__": _safe_import,
        "result": None, # Default to None to detect missing assignments
        "_getitem_": default_guarded_getitem,      
        "_getiter_": default_guarded_getiter,      
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "len": len
    })
    
    try:
        byte_code = compile_restricted(clean_code, filename='<inline>', mode='exec')
        exec(byte_code, safe_env)
        
        # Check if the AI actually assigned a value
        final_result = safe_env.get('result')
        if final_result is None:
            return "Execution Success, but you forgot to assign the answer to the 'result' variable."
        return str(final_result)
        
    except Exception as e:
        return f"Execution Error: {str(e)}"