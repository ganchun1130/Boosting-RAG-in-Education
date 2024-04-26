import csv
import json
import time
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS

from config import *
from Chat.MyLLM import ChatGLM

def get_query_from_csv(csv_filename):
    # 初始化一个空列表来存储提取的内容
    extracted_content = []

    # 打开CSV文件并读取内容
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)

        # 跳过标题行（如果有）
        next(csv_reader, None)

        # 遍历CSV文件中的每一行
        for row in csv_reader:
            if len(row) == 2:
                # 提取每行的第一个元素（逗号前面的内容）
                extracted_content.append(row[0])

    # 打印提取的内容
    # for item in extracted_content:
    #     print(item)

    return extracted_content


def get_answer_and_context_from_llm(query_list, llm, embeddings, top_k):
    query_num = len(query_list)
    context_from_retriever = []
    answer_from_llm = []

    prompt_template = """你是一个真诚且友善的智能助手，你现在服务于一个名为小雅的智能教育系统。
                    现在有用户向你询问系统的某些使用方法，请你基于以下已知内容来清楚地、有条理地以及精确地回答用户的问题。
                    如果你无法从中得到答案，请说没有提供足够的相关信息。不允许在答案中添加编造成分。另外，答案请使用中文。

                    用户问题：{question},已知内容：{context}。
                    """

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    vector_store = FAISS.load_local(os.path.join(VECTOR_STORE_PATH, VECTOR_STORE_NAME), embeddings)

    knowledge_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": top_k}),
        prompt=prompt)
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}")
    knowledge_chain.return_source_documents = True

    for i in range(query_num):
        query = query_list[i]

        if i%10==0:
            time.sleep(5)

        result = knowledge_chain({"query": query})

        answer, docs = result["results"], result["source_documents"]

        # Print the results
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
        # 将大模型生成的答案加入到列表中
        context_from_retriever.append(list(set([document.page_content for document in docs])))
        answer_from_llm.append(result["results"])

    return context_from_retriever, answer_from_llm


def write_data_to_json(answers_list, contexts_list, json_filename):


    # 假设这是您的answer和context列表
    answers = answers_list
    contexts = contexts_list

    # 初始化一个空列表来存储将要写入的JSON对象
    updated_json_data = []

    # 读取JSON文件
    with open(json_filename, 'r', encoding='utf-8') as jsonfile:
        json_data = json.load(jsonfile)

    # 确保answer和context列表的长度与JSON数据列表中的条目数相匹配
    if len(answers) != len(json_data) or len(contexts) != len(json_data):
        raise ValueError("The length of answers and contexts must match the number of entries in the JSON data.")

    # 遍历JSON数据列表，并为每个条目添加answer和contexts字段
    for i in range(len(json_data)):
        current_answer = answers[i]
        current_contexts = contexts[i]
        json_data[i].update({
            'answer': current_answer,
            'contexts': current_contexts
        })

    # 写入更新后的JSON文件
    with open(json_filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_data, jsonfile, ensure_ascii=False, indent=4)

    print(f'Done. Data updated and written to {json_filename}')


if __name__ == "__main__":
    # CSV文件名
    csv_filename = "/usr/local/TFBOYS/gc/NLP/LLM_RAG_API/data/collection_of_user_problems_202309-202312.csv"
    json_filename = "/usr/local/TFBOYS/gc/NLP/LLM_RAG_API/data/naive_eval_data.json"
    llm = ChatGLM()
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[init_embedding_model])

    query_list = get_query_from_csv(csv_filename=csv_filename)

    context_from_retriever,answer_from_llm = get_answer_and_context_from_llm(query_list=query_list,
                                                                            llm=llm,
                                                                            embeddings=embeddings,
                                                                            top_k=3
                                                                            )
    print(context_from_retriever)
    print(answer_from_llm)

    write_data_to_json(answers_list=answer_from_llm, contexts_list=context_from_retriever, json_filename=json_filename)