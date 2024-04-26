import abc
import os
from typing import Type
from langchain.tools import BaseTool
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import faiss
from pydantic import BaseModel, Field

from Chat.MyLLM import ChatGLM
from config import (VECTOR_STORE_PATH,
                    VECTOR_STORE_NAME,
                    embedding_model_dict,
                    init_embedding_model,
                    KNOWLEDGE_FILE_PATH,
                    KNOWLEDGE_FILE2)
from RAG.retrieval.retriever import small2big_contextual_compression_retriever


class RetrievalInput(BaseModel):
    query: str = Field(description="Query to perform retrieval")


class RefineCoTAgent(BaseTool, abc.ABC):
    name = "RefineCoTAgent"
    description = """使用LLM生成CoT，根据用户问题、任务提示和检索信息。
    若信息不足，LLM指定新的查询进行二次检索，然后更新CoT。LLM用最终CoT回答问题。"""
    args_schema: Type[BaseModel] = RetrievalInput

    def __init__(self):
        super().__init__()

    def _generate_cot_once(self, llm, user_question, retrieved_info=None):
        # 生成初步的链式思维（CoT）
        prompt = \
            f"""这是用户的问题：{user_question} ；
                这是检索到的内容：{retrieved_info}；

                请仔细阅读上述用户问题和检索到的内容。
                首先评估检索到的信息是否足够完整和准确，以回答用户的问题。
                如果检索到的信息足以全面回答，请发挥你的逻辑推理能力，一步一步地思考并构建具有结构化的答案。
                如果检索到的信息不足以回答问题的各个方面，或者存在不确定性，请在答案开头明确表示“需要更多信息”，并尽可能提供基于当前信息的初步分析。              
                """
        ans = llm._call(prompt)
        return ans

    def _generate_cot_twice(self, llm, user_question, retrieved_info=None):
        prompt = \
            f"""这是用户的问题：{user_question} ；
                这是检索到的内容：{str(retrieved_info)}；

                请仔细阅读上述用户问题和检索到的内容。
                然后发挥你的逻辑推理能力，一步一步地思考；清楚地、有条理地以及精确地回答用户的问题。

            """
        print(user_question)
        ans = llm._call(prompt)
        return ans

    def _run(self, query: str) -> str:

        llm = ChatGLM()
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[init_embedding_model])
        # 根据query进行矢量数据库检索


        # 使用FAISS类来创建一个向量存储
        vector_store = FAISS.load_local(os.path.join(VECTOR_STORE_PATH, VECTOR_STORE_NAME), embeddings)
        retriever = small2big_contextual_compression_retriever(vector_store=vector_store,
                                                               embeddings=embeddings,
                                                               top_k=4,
                                                               file_path=os.path.join(KNOWLEDGE_FILE_PATH,
                                                                                                KNOWLEDGE_FILE2),
                                                               )
        docs1 = retriever.get_relevant_documents(query)
        retrieved_info = []
        for doc1 in docs1:
            retrieved_info.append(doc1.page_content)
        print(retrieved_info)

        # 生成初步的链式思维（CoT）
        ans = self._generate_cot_once(llm, query, retrieved_info=retrieved_info)
        print(f"这是LLM第一次回答：{ans}")

        # 判断是否需要继续检索
        if "需要更多信息" in ans:
            # 提取需要检索的query
            new_query = ans.split("需要更多信息: ")[-1]
            print(new_query)
            # 进行第二次检索

            additional_info = []
            docs2 = retriever.get_relevant_documents(new_query)
            for doc2 in docs2:
                additional_info.append(doc2.page_content)
            additional_info.extend(retrieved_info)

            if additional_info:
                # 生成修正后的链式思维（CoT）
                ans = self._generate_cot_twice(llm, query, retrieved_info=additional_info)
                print(f"这是LLM第二次回答：{ans}")

        return ans