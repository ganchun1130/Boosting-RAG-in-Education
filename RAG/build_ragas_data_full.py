import csv
import json
import os
import time
import logging
import datetime

from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from llama_index.embeddings import resolve_embed_model
from llama_index.llms import OpenAI
from llama_index.response.notebook_utils import display_source_node

from config import *
from Chat.MyLLM import ChatGLM, Qwen7B, Qwen14B
from RAG.rag_functions import (
    naive_rag,
    small2big_llama_index,
    small2big_langchain,
    contextual_compression,
    multi_query_langchain,
    Boost_RAG_with_TCoT_brage
)
from RAG.retrieval.retriever import *
from config import embedding_model_dict, bge_embedding_model, text2vector_embedding_model


def get_question_and_truth_from_csv(csv_filename):
    # 初始化一个空列表来存储提取的内容
    extracted_question = []
    extracted_truth = []

    # 打开CSV文件并读取内容
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)

        # 跳过标题行（如果有）
        next(csv_reader, None)

        # 遍历CSV文件中的每一行
        for row in csv_reader:
            if len(row) == 2:
                # 提取每行的第一个元素（逗号前面的内容）
                extracted_question.append(row[0])
                extracted_truth.append(row[1])

    # 打印提取的内容
    if len(extracted_question) == len(extracted_truth):
        print(f"提取到{len(extracted_question)}条数据")
        for i in range(len(extracted_question)):
            print(f"问题：{extracted_question[i]}")
            print(f"答案：{extracted_truth[i]}")
    else:
        print("提取数据出错，请检查CSV文件")

    return extracted_question, extracted_truth


def get_answer_and_context_from_llm_langchain(query, knowledge_chain):
    context_from_retriever = []
    answer_from_llm = []

    result = knowledge_chain({"query": query})
    answer, docs = result["result"], result["source_documents"]

    # Print the result
    print("\n\n> Question:")
    print(query)
    print("\n> Answer:")
    print(answer)

    for document in docs:
        print("\n> source document：" + document.metadata["source"] + ":")
        print(document.page_content)
        # 将检索到的所有内容document.page_content，作为一个列表加入到context_from_retriever中
    context_from_retriever.append(list(set([document.page_content for document in docs])))
    # 将大模型生成的答案加入到列表中
    answer_from_llm.append(result["result"])
    return context_from_retriever, answer_from_llm


def get_answer_and_context_from_llm_llama_index(query, query_engine_chunk, retriever_chunk):
    nodes = retriever_chunk.retrieve(query)
    answer_from_llm = []
    context_from_retriever = []
    # for node in nodes:
    #     display_source_node(node, source_length=2000)
    context_from_retriever.append(list(set([node.get_content() for node in nodes])))
    answer_from_llm.append(str(query_engine_chunk.query(query)))

    print(answer_from_llm)
    print(context_from_retriever)

    return context_from_retriever, answer_from_llm


def write_to_json(question, answer, contexts, ground_truth, json_filename):
    # 读取已有的JSON文件
    try:
        with open(json_filename, 'r', encoding='utf-8') as jsonfile:
            json_data = json.load(jsonfile)
    except FileNotFoundError:
        json_data = []

    # 添加新的数据
    json_data.append({
        'question': question,
        'answer': answer,
        'contexts': contexts,
        'ground_truths': [ground_truth]
    })

    # 写入JSON文件
    with open(json_filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_data, jsonfile, ensure_ascii=False, indent=4)

    print(f'Done. Data written to {json_filename}')


def build_ragas_data_full_langchain(csv_filename, top_k, json_filename, rag_function, rerank: bool,
                                    knowledge_file_path):
    # 读取CSV文件中的问题和答案
    question_list, truth_list = get_question_and_truth_from_csv(csv_filename)

    # 定义 llm, knowledge_chain, embeddings
    llm1 = Qwen7B()
    llm2 = Qwen14B()
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[text2vector_embedding_model])

    print(f"you are using the {rag_function} function!")

    if 'naive' in rag_function.lower():
        knowledge_chain = naive_rag(llm=llm2, embeddings=embeddings, top_k=top_k, rerank=rerank)

    elif 'rag_fusion' in rag_function.lower():
        knowledge_chain = multi_query_langchain(llm1=llm1,
                                                llm2=llm2,
                                                embeddings=embeddings,
                                                top_k=top_k,
                                                retriever_type=rag_function,
                                                file_path=knowledge_file_path,
                                                rerank=rerank)

    elif 'contextual' or 'compression' in rag_function.lower():
        knowledge_chain = contextual_compression(llm=llm2,
                                                 embeddings=embeddings,
                                                 top_k=top_k,
                                                 retriever_type=rag_function,
                                                 rerank=rerank,
                                                 file_path=knowledge_file_path)
    elif 'small2big' in rag_function.lower():
        knowledge_chain = small2big_langchain(llm=llm2,
                                              embeddings=embeddings,
                                              top_k=top_k,
                                              file_path=knowledge_file_path,
                                              rerank=rerank)

    else:
        print(f"""we can't find the function which name is {rag_function}!\n 
        please input correct function name next time!\n 
        However, to make you fell happy, we provide the naive rag function to run your code!""")
        knowledge_chain = naive_rag(llm=llm2,
                                    embeddings=embeddings,
                                    top_k=top_k,
                                    rerank=rerank)
        # 遍历问题列表
    for i in range(len(question_list)):

        if i % 5 == 0:
            time.sleep(5)

        query = question_list[i]
        ground_truth = truth_list[i]
        print(f"正在处理第{i + 1}条数据")

        # 调用LLM，获取答案和上下文
        context_from_retriever, answer_from_llm = get_answer_and_context_from_llm_langchain(query, knowledge_chain)
        # contexts = retriever.get_relevant_documents(query)
        # context_from_retriever = [list(set([document.page_content for document in contexts]))]

        if len(answer_from_llm) == len(context_from_retriever) == 1:
            time.sleep(2)
            # 写入JSON文件
            write_to_json(query, answer_from_llm[0], context_from_retriever[0], ground_truth, json_filename)


def build_ragas_data_full_CoT(csv_filename,
                              top_k,
                              json_filename,
                              rerank: bool,
                              rag_function: str=None,
                              knowledge_file_path=None,
                              similarity_threshold=0.5
                              ):
    # 读取CSV文件中的问题和答案
    question_list, truth_list = get_question_and_truth_from_csv(csv_filename)

    # 定义 llm, knowledge_chain, embeddings
    llm1 = Qwen7B()
    llm2 = Qwen14B()
    # embeddings1 = HuggingFaceBgeEmbeddings(model_name=embedding_model_dict[bge_embedding_model])
    embeddings2 = HuggingFaceEmbeddings(model_name=embedding_model_dict[text2vector_embedding_model])

    vector_store = FAISS.load_local(os.path.join(VECTOR_STORE_PATH, VECTOR_STORE_NAME2), embeddings2)

    if "small2big" and "contextual_compression" in rag_function.lower():
        retriever = small2big_contextual_compression_retriever(file_path=knowledge_file_path,
                                                               embeddings=embeddings2,
                                                               top_k=top_k,
                                                               vector_store=vector_store,
                                                               rerank=rerank,
                                                               similarity_threshold=similarity_threshold)
    elif "rag_fusion" in rag_function.lower():

        if "small2big" in rag_function.lower():
            temp_retriever = small2big_retriever(vector_store=vector_store,
                                                 top_k=top_k,
                                                 rerank=rerank,
                                                 file_path=knowledge_file_path)
        elif "contextual_compression" in rag_function.lower():
            temp_retriever = contextual_compression_retriever(vector_store=vector_store,
                                                              top_k=top_k,
                                                              rerank=rerank,
                                                              embeddings=embeddings2)
        else:
            temp_retriever = naive_retriever(vector_store=vector_store,
                                             top_k=top_k,
                                             rerank=rerank)
        retriever = multi_query_retriever(llm=llm1,
                                          retrievers=temp_retriever,
                                          top_k=top_k,
                                          rerank=rerank)
    elif "all" in rag_function.lower():
        retriever = super_retriever_msc(vector_store=vector_store,
                                        llm1=llm1,
                                        embeddings=embeddings2,
                                        top_k=top_k,
                                        rerank=rerank,
                                        similarity_threshold=similarity_threshold,
                                        file_path=knowledge_file_path)
    elif "small2big" in rag_function.lower():
        retriever = small2big_retriever(vector_store=vector_store,
                                        top_k=top_k,
                                        rerank=rerank,
                                        file_path=knowledge_file_path)
    elif "contextual_compression" in rag_function.lower():
        retriever = contextual_compression_retriever(vector_store=vector_store,
                                                     top_k=top_k,
                                                     rerank=rerank,
                                                     embeddings=embeddings2)
    else:
        retriever = naive_retriever(vector_store=vector_store,
                                    top_k=top_k,
                                    rerank=rerank)

    for i in range(len(question_list)):

        if i % 5 == 0:
            time.sleep(5)

        query = question_list[i]
        ground_truth = truth_list[i]
        print(f"正在处理第{i + 1}条数据")

        # 调用LLM，获取答案和上下文
        answer_from_llm, context_from_retriever = Boost_RAG_with_TCoT_brage(retriever=retriever,
                                                                            llm=llm2,
                                                                            query=query
                                                                            )
        # if len(context_from_retriever) == 1:
        # 写入JSON文件
        time.sleep(2)
        write_to_json(query, answer_from_llm, context_from_retriever, ground_truth, json_filename)


def build_ragas_data_full_llama_index(csv_filename, knowledge_file_path, top_k, json_filename):
    # 读取CSV文件中的问题和答案
    question_list, truth_list = get_question_and_truth_from_csv(csv_filename)

    # 定义 llm, knowledge_chain, embeddings
    embed_model = resolve_embed_model(f"local:{embedding_model_dict[text2vector_embedding_model]}")
    llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, api_base=ChatGLM_BASE_URL)

    query_engine_chunk, retriever_chunk = small2big_llama_index(knowledge_file_path=knowledge_file_path, llm=llm,
                                                                embeddings=embed_model, top_k=top_k)

    # 遍历问题列表
    for i in range(len(question_list)):

        query = question_list[i]
        ground_truth = truth_list[i]
        print(f"正在处理第{i + 1}条数据")

        # 调用LLM，获取答案和上下文
        context_from_retriever, answer_from_llm = get_answer_and_context_from_llm_llama_index(query=query,
                                                                                              query_engine_chunk=query_engine_chunk,
                                                                                              retriever_chunk=retriever_chunk)
        if len(answer_from_llm) == len(context_from_retriever) == 1:
            # 写入JSON文件
            write_to_json(query, answer_from_llm[0], context_from_retriever[0], ground_truth, json_filename)


if __name__ == '__main__':

    # 获取当前时间并格式化为字符串
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_filename = f'log/output_{current_time}.log'

    # 配置logging
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
        datefmt='%Y-%m-%d %H:%M:%S',  # 设置日期格式
        handlers=[
            logging.StreamHandler(),  # 将日志输出到控制台
            logging.FileHandler(log_filename)  # 将日志输出到文件
        ]
    )

    csv_filename = "/usr/local/TFBOYS/gc/NLP/LLM_RAG_API/Evaluation/data/collection_of_user_problems_100_202309-202312.csv"
    knowledge_file_path = "/usr/local/TFBOYS/gc/NLP/ChatXIaoYa_chatglm/knowledge_file"
    knowledge_file1 = "promax_data_zh_processed2.pdf"
    knowledge_file2 = "promax_data_zh_processed_after_chatglm3.pdf"

    json_file_root_path = "/usr/local/TFBOYS/gc/NLP/LLM_RAG_API/Evaluation/data/"
    json_filename1 = "naive_eval_data_updated.json"
    

    # 下面是测试的
    json_filename = "test"


    logging.info("start!")
    build_ragas_data_full_langchain(csv_filename,
                                    top_k=5,
                                    json_filename=os.path.join(json_file_root_path, json_filename1),
                                    rag_function="rag_fusion&small",   # 如果你想使用rag_fusion,请将rag_fusion放在最前面
                                    rerank=True,
                                    knowledge_file_path=os.path.join(knowledge_file_path, knowledge_file1)
                                    )
    # build_ragas_data_full_llama_index(csv_filename=csv_filename, top_k=4, json_filename=json_filename2, knowledge_file_path=knowledge_file_path)
    # build_ragas_data_full_CoT(csv_filename,
    #                           top_k=4,
    #                           json_filename=os.path.join(json_file_root_path, json_filename16),
    #                           knowledge_file_path=os.path.join(knowledge_file_path, knowledge_file1),
    #                           rerank=True,
    #                           rag_function="all",
    #                           )
    logging.info("ByeBye!")
