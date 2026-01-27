from typing import Annotated, TypedDict, List, Optional
from langgraph.graph.message import add_messages

class RLMState(TypedDict):
    # The conversation history
    messages: Annotated[list, add_messages]
    # The snippet of the manual the agent is currently looking at
    current_context: str
    # Recursion depth to prevent infinite loops (Bank safety)
    depth: int