from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages

# Reducer to merge new search logs into the existing list
def update_history(left: List[str], right: List[str]) -> List[str]:
    return left + right

class RLMState(TypedDict):
    messages: Annotated[list, add_messages]
    current_context: str
    depth: int  # Current loop count
    search_history: Annotated[List[str], update_history]
    confidence: float # New: 0.0 to 1.0