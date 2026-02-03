import asyncio
from langgraph_sdk import get_client

async def run_test():
    # 1. Initialize the client (pointing to your local dev server)
    client = get_client(url="http://localhost:2024")
    
    # 2. Select your assistant (the 'agent' ID from your config)
    assistant_id = "agent"
    
    # 3. Define the input state
    # We include search_history and depth as per your RLMState [cite: 2026-01-27]
    input_state = {
        "messages": [
            {
                "role": "user", 
                "content": "Find the Risk Management Framework section... extract the Current Risk Situation. Perform the import functions with what you already have acces to"
            }
        ],
        "depth": 0,
        "search_history": [],
        "confidence": 0.0
    }
    
    print("--- Starting Stream ---")
    
   # 4. Stream the results
    async for event in client.runs.stream(
        None,  # Thread ID
        assistant_id,
        input=input_state,
        stream_mode="values",
    ):
        # FIX: Check if "messages" key exists in the data dictionary
        if event.data and "messages" in event.data:
            last_msg = event.data["messages"][-1]
            history = event.data.get("search_history", [])
            
            # Extract content correctly (handles both dict and object formats)
            content = ""
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
            else:
                content = getattr(last_msg, "content", "")

            print(f"\n[Node Event]: {event.event}")
            print(f"[Content]: {content[:200]}...") # Truncated for readability
            print(f"[History Count]: {len(history)}")
        else:
            # This handles the metadata/start events silently
            print(f"--- Processing: {event.event} ---")

if __name__ == "__main__":
    asyncio.run(run_test())