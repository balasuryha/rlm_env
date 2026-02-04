import asyncio
from langgraph_sdk import get_client

async def run_test():
    # 1. Initialize the client (pointing to your local dev server)
    client = get_client(url="http://localhost:2024")
    
    # 2. Select your assistant (the 'agent' ID from your config)
    assistant_id = "agent"
    
    # 3. Define the input state
    input_state = {
        "messages": [
            {
                "role": "user", 
                "content": "Find the financial review section"
            }
        ],
        "depth": 0,
        "search_history": [],
        "confidence": 0.0
    }
    
    print("=" * 80)
    print("STARTING RECURSIVE SEARCH TEST")
    print("=" * 80)
    
    # 4. Stream the results
    async for event in client.runs.stream(
        None,  # Thread ID
        assistant_id,
        input=input_state,
        stream_mode="values",
    ):
        # Check if "messages" key exists in the data dictionary
        if event.data and "messages" in event.data:
            last_msg = event.data["messages"][-1]
            history = event.data.get("search_history", [])
            depth = event.data.get("depth", 0)
            confidence = event.data.get("confidence", 0.0)
            
            # Extract content correctly (handles both dict and object formats)
            content = ""
            role = "unknown"
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
                role = last_msg.get("role", "unknown")
            else:
                content = getattr(last_msg, "content", "")
                role = getattr(last_msg, "type", "unknown")

            print("\n" + "=" * 80)
            print(f"[EVENT]: {event.event}")
            print(f"[ROLE]: {role}")
            print(f"[DEPTH]: {depth}")
            print(f"[CONFIDENCE]: {confidence}")
            print(f"[HISTORY COUNT]: {len(history)}")
            if history:
                print(f"[LAST HISTORY]: {history[-1]}")
            print("-" * 80)
            print(f"[CONTENT]:")
            print(content)  # Full content, no truncation
            print("=" * 80)
            
            # Small delay to ensure output is visible
            await asyncio.sleep(0.1)
        else:
            # This handles the metadata/start events
            print(f"\n--- Processing: {event.event} ---")

    print("\n" + "=" * 80)
    print("STREAM COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_test())