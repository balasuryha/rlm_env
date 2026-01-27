import os
from langgraph_sdk import get_sync_client
from dotenv import load_dotenv

load_dotenv()

# 1. Connect to your running 'langgraph dev' server
# The port is usually 2024
client = get_sync_client(url="http://localhost:2024")

def run_test():
    # 'agent' is the name from your langgraph.json
    print("🤖 Sending request to Banking Agent...")
    
    # Start a threadless run
    input_data = {
        "messages": [{"role": "user", "content": "What are the fees for international transfers?"}]
    }
    
    # Use stream to see the recursive steps
    for chunk in client.runs.stream(None, "agent", input=input_data):
        if chunk.event == "values":
            # This shows the current state (messages)
            last_msg = chunk.data["messages"][-1]
            print(f"\n[{chunk.event}] {last_msg['type'].upper()}: {last_msg['content'][:100]}...")
        elif chunk.event == "metadata":
            print(f"📍 Node: {chunk.data.get('node')}")

if __name__ == "__main__":
    run_test()