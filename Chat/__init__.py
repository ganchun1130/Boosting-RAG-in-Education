# curl http://172.0.101.52:9000/v1/chat/completions \
#    -H "Content-Type: application/json" \
#    -d '{
#       "model": "qwen",
#       "messages": [{"role": "user", "content": "Say this is a test!"}],
#       "temperature": 0.7
#     }'
# docker run --name one-api -d --restart always --privileged=true -p 3000:3000 -e TZ=Asia/Shanghai -v /home/ubuntu/data/one-api:/data ghcr.io/songquanpeng/one-api