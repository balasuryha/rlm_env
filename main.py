# main.py
from src.agent.graph import graph
from dotenv import load_dotenv

load_dotenv()

inputs = {"messages": [("user", "What is the penalty for late payments on a Home Loan?")], "depth": 0}

for event in graph.stream(inputs):
    print(event)