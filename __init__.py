# curl http://localhost:11434/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "qwen:0.5b",
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant."
#             },
#             {
#                 "role": "user",
#                 "content": "Hello!"
#             }
#         ]
#     }'

