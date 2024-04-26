import openai
# import http.client
# http.client.HTTPConnection._http_vsn = 10
# http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

# qwen1.5系列模型还未能找到官方调用api的代码文件，以下代码是基于ollama的调用方式
from openai import OpenAI

client = OpenAI(
    base_url = 'http://172.0.101.52:11434/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
  model="qwen:14b",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The LA Dodgers won in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
print(response.choices[0].message.content)