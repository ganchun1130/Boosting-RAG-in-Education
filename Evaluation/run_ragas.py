import datetime
import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
from Chat.MyLLM import ChatGLM, Qwen14B,  Qwen7B
from config import *
import os

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    # answer_similarity,
    # answer_correctness,
    # AspectCritique,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision
)
from ragas.metrics.critique import (
    harmfulness,
    # maliciousness,
    # coherence,
    # correctness,
    # conciseness
)
from ragas import evaluate
from ragas.embeddings import HuggingfaceEmbeddings

from Evaluation import reports_functions
from Evaluation.MyRagasLLM import MyLLM_ChatGLM, MyLLM_Qwen


def eva(llm, custom_prefix, file_path):
    # use the Ragas LangchainLLM wrapper to create a RagasLLM instance
    # llm = MyLLM(llm=ChatGLM3())
    ans_relevancy = AnswerRelevancy(embeddings=HuggingFaceEmbeddings(model_name=embedding_model_dict[text2vector_embedding_model]))
    # change the LLM
    faithfulness.llm = llm
    ans_relevancy.llm = llm
    # answer_similarity.llm = llm
    # answer_correctness.llm = llm
    context_relevancy.llm = llm
    context_precision.llm = llm
    context_recall.llm = llm
    # harmfulness.llm = llm
    # maliciousness.llm = llm
    # coherence.llm = llm
    # correctness.llm = llm
    # conciseness.llm = llm

    datasets = Dataset.from_json(file_path)

    # answer_similarity.threshold = None  # 不使用阈值，可直接记录相似度
    # answer_correctness.answer_similarity.threshold = None  # 同上

    # my_aspect = AspectCritique(name="children", definition="Is the submission safe to children?") #自定义一个aspect

    result = evaluate(datasets, metrics=[ans_relevancy,
                                         faithfulness,
                                         # answer_similarity,
                                         # answer_correctness,
                                         context_relevancy,
                                         context_precision,
                                         context_recall,
                                         # harmfulness,
                                         # maliciousness,
                                         # coherence,
                                         # correctness,
                                         # conciseness,
                                         # my_aspect
                                         ])
    # print(results)
    reports_functions.gen_report(result, custom_prefix=custom_prefix)
# /root/anaconda3/envs/torch2/lib/python3.11/site-packages/ragas/metrics

if __name__ == '__main__':

    # 获取当前时间并格式化为字符串
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_filename = f'log/eval_output_{current_time}.log'

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

    llm = MyLLM_Qwen(llm=Qwen14B())
    custom_prefix = "cc_s2b_14_small"
    # custom_prefix = "tt"
    file_path = os.path.join(ragas_dataset_base_path, ragas_dataset_name[custom_prefix])
    print(file_path)
    eva(llm=llm, custom_prefix=custom_prefix, file_path=file_path)
