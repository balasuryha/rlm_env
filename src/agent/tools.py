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
    MANUAL_CONTENT = pymupdf4llm.to_markdown("data/six-annual-report-2024-en.pdf")
except Exception as e:
    MANUAL_CONTENT = "Error loading PDF: " + str(e)

# Safe import handler
def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    Allow only 're' and 'json' to be imported inside sandbox.
    """
    if name in ("re", "json"):
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Module '{name}' is restricted in sandbox.")

@tool
def recursive_document_search(code: str):
    """
    Safely execute Python code in sandbox. 
    Available variables:
        - doc: preloaded document as string (markdown)
        - MANUAL_CONTENT: same as doc
        - result: must assign output here
    Only allowed imports: re, json
    """
    # Clean AI markdown code blocks if present
    clean_code = re.sub(r'^```python\n|```$', '', code, flags=re.MULTILINE).strip()

    logger.info("Executing sandbox code:\n%s", clean_code)

    # Safe environment
    safe_env = safe_globals.copy()
    safe_env.update({
        "doc": MANUAL_CONTENT,
        "document": MANUAL_CONTENT,
        "MANUAL_CONTENT": MANUAL_CONTENT,
        "re": re,
        "json": json,
        "result": None,

        # Guards for RestrictedPython
        "__import__": _safe_import,
        "_getattr_": getattr,
        "_write_": full_write_guard,
        "_getitem_": default_guarded_getitem,
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_print_": PrintCollector,

        # basic built-ins
        "len": len, "str": str, "int": int, "list": list, "dict": dict
    })

    try:
        byte_code = compile_restricted(clean_code, filename='<inline>', mode='exec')
        exec(byte_code, safe_env)
        final_result = safe_env.get('result')
        return str(final_result) if final_result is not None else "Success, but 'result' not set."
    except Exception as e:
        return f"Execution Error: {str(e)}"
