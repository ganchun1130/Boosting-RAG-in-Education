# curl http://172.0.101.52:3000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -H "Authorization: Bearer sk-c4GOPTvMpPQojDDL95F89aE0F8A344F7A2514d943bDe2714" \
#     -d '{
#            "model": "ChatGLM3-6B-32K",
#            "messages": [{"role": "user", "content": "Say this is a test!"}],
#            "temperature": 0.7
#          }'
#
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

