import openai
# import http.client
# http.client.HTTPConnection._http_vsn = 10
# http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

from openai import OpenAI

# remove the following line when you are ready to use the API
api_key = ""
base_url1 = ""

client = OpenAI(base_url=base_url1, api_key="EMPTY", )

def simple_chat(use_stream=True):
    messages = [
        # {
        #     "role": "system",
        #     "content": "You are Qwen, a large language model trained by ali. Follow the user's "
        #                "instructions carefully. Respond using markdown.",
        # },
        {
            "role": "user",
            "content": "你好，你是谁？"
        }
    ]
    response = client.chat.completions.create(
        model="qwen",
        messages=messages,
        stream=use_stream,
        max_tokens=256,
        temperature=0.8,
        presence_penalty=1.1,
        top_p=0.8)
    print(response)

if __name__ == "__main__":
    simple_chat(use_stream=False)
    # simple_chat(use_stream=True)
    # embedding()
    # function_chat()