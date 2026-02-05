prompt:

system_content = (
    f"Attempt {current_depth + 1}/{MAX_RECURSION_DEPTH}. "
    "Document loaded as `doc` (markdown).\n\n"
    "STEP 1: Write a Python code block using `re` or `json` to search the `doc` variable. "
    "Assign your findings to the variable `result`. Use a real regex pattern based on the user's query.\n\n"
    "STEP 2: Once you have results from a previous code execution, synthesize the final answer.\n\n"
    "CRITICAL:\n"
    "- If you are writing code to search, set 'confidence' to 0.0 in your JSON.\n"
    "- ONLY set 'confidence' to 1.0 once you have seen a 'CODE_EXECUTION_RESULT' that contains the answer.\n\n"
    "FORMAT REQUIRED:\n"
    "```python\n"
    "import re\n"
    "result = re.findall(r'Actual Query Pattern', doc)\n"
    "```\n\n"
    '{"answer": "Providing search results...", "confidence": 0.0}'
)