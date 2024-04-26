import os
import re
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import gradio as gr
import sentence_transformers

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever, MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from MyLLM import ChatGLM, Qwen
from config import *

embedding_model_dict = embedding_model_dict
llm_model_dict = llm_model_dict
EMBEDDING_DEVICE = EMBEDDING_DEVICE
LLM_DEVICE = LLM_DEVICE
num_gpus = num_gpus


llm_model_list = []
for i in llm_model_dict:
    for j in llm_model_dict[i]:
        llm_model_list.append(j)

prompt_template = """你是一个真诚且友善的智能助手，你现在服务于一个名为小雅的智能教育系统。
                            现在有用户向你询问系统的某些使用方法，请你基于以下已知内容来清楚地、有条理地以及精确地回答用户的问题。
                            如果你无法从中得到答案，请说没有提供足够的相关信息。不允许在答案中添加编造成分。另外，答案请使用中文。

                            用户问题：{question},已知内容：{context}。
                            """

prompt = PromptTemplate(template=prompt_template,
                        input_variables=["context", "question"])

model_status = """初始模型已加载成功！可以开始对话！"""


def clear_session():
    return '', None


def predict(query,
            top_k,
            temperature,
            history=None):
    if history == None:
        history = []

    # 定义 llm, knowledge_chain, embeddings
    llm = ChatGLM(temperature=temperature)
    # qwen = Qwen(temperature=temperature)
    print("LLM 加载成功！")
    encode_kwargs = {'normalize_embeddings': True}  # 设置为True以计算余弦相似度
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[init_embedding_model], model_kwargs={'device': 'cpu'}, encode_kwargs=encode_kwargs)

    print("embedding 加载成功！")

    vector_store = FAISS.load_local(os.path.join(VECTOR_STORE_PATH, VECTOR_STORE_NAME), embeddings)

    knowledge_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": top_k}),
        prompt=prompt)
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}")
    knowledge_chain.return_source_documents = True

    result = knowledge_chain({"query": query})
    answer, docs = result["result"], result["source_documents"]

    # Print the result
    print("\n\n> Question:")
    print(query)
    print("\n> Answer:")
    print(answer)

    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)

    history.append((query, answer))
    return '', history, history


if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:
        gr.Markdown("""<h1><center>LangChain-ChatGLM-XiaoYa-WebUI</center></h1>
        <center><font size=3>
        本项目基于LangChain和大型语言模型系列模型, 提供基于本地知识的自动问答应用. 
        </center></font>
        """)
        model_status = gr.State(model_status)
        with gr.Row():
            with gr.Column(scale=1):
                model_choose = gr.Accordion("模型选择")
                with model_choose:
                    large_language_model = gr.Dropdown(
                        llm_model_list,
                        label="large language model",
                        value=init_llm)

                    embedding_model = gr.Dropdown(list(
                        embedding_model_dict.keys()),
                        label="Embedding model",
                        value=init_embedding_model)
                    # load_model_button = gr.Button("重新加载模型")
                model_argument = gr.Accordion("模型参数配置")
                with model_argument:
                    top_k = gr.Slider(1,
                                      10,
                                      value=6,
                                      step=1,
                                      label="vector search top k",
                                      interactive=True)

                    history_len = gr.Slider(0,
                                            5,
                                            value=3,
                                            step=1,
                                            label="history len",
                                            interactive=True)

                    temperature = gr.Slider(0,
                                            1,
                                            value=0.01,
                                            step=0.01,
                                            label="temperature",
                                            interactive=True)
                    top_p = gr.Slider(0,
                                      1,
                                      value=0.9,
                                      step=0.1,
                                      label="top_p",
                                      interactive=True)

                file = gr.File(label='请上传知识库文件，文件格式只支持pdf',
                               file_types=['.pdf'])

                # init_vs = gr.Button("知识库文件向量化")

                use_web = gr.Radio(["True", "False"],
                                   label="Web Search",
                                   value="False")

            with gr.Column(scale=4):
                chatbot = gr.Chatbot([[None, model_status.value]],
                                     label='ChatLLM').style(height=750)
                message = gr.Textbox(label='请输入问题~')
                state = gr.State()

                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    send = gr.Button("🚀 发送")

            # load_model_button.click(
            #     reinit_model,
            #     show_progress=True,
            #     inputs=[large_language_model, embedding_model, chatbot],
            #     outputs=chatbot,
            # )
            # init_vs.click(
            #     init_vector_store,
            #     show_progress=True,
            #     inputs=[file],
            #     outputs=[],
            # )

            send.click(predict,
                       inputs=[
                           message, top_k, temperature
                       ],
                       outputs=[message, chatbot, state])
            clear_history.click(fn=clear_session,
                                inputs=[],
                                outputs=[chatbot, state],
                                queue=False)

            message.submit(predict,
                           inputs=[
                               message, top_k, temperature
                           ],
                           outputs=[message, chatbot, state])
        gr.Markdown("""提醒：<br>
        1. 使用时请先上传自己的知识文件，并且文件中不含某些特殊字符，否则将返回error. <br>
        2. 有任何使用问题，请通过[Github Issue区](https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/issues)进行反馈. <br>
        """)
    # threads to consume the request
    demo.queue(concurrency_count=3) \
        .launch(server_name='0.0.0.0',
                # ip for listening, 0.0.0.0 for every inbound traffic, 127.0.0.1 for local inbound
                server_port=7860,  # the port for listening
                show_api=False,  # if display the api document
                share=False,  # if register a public url
                inbrowser=False)  # if browser would be open automatically
