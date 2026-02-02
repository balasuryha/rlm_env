test - curl -X POST http://localhost:2024/runs/stream \
     -H "Content-Type: application/json" \
     -d '{
       "assistant_id": "agent",
       "input": {
         "messages": [
           {
             "role": "user",
             "content": "Find the Risk Management Framework section... extract the Current Risk Situation."
           }
         ],
         "depth": 0,
         "confidence": 0.0,
         "search_history": [],
         "current_context": ""
       },
       "stream_mode": "values"
     }'
