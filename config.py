import os
import torch

# device config
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = torch.cuda.device_count()


MODEL_CACHE_PATH = "/cache"

# vector storage config
VECTOR_STORE_NAME1 = ''
VECTOR_STORE_NAME2 = ""
VECTOR_STORE_NAME3 = ''
VECTOR_STORE_NAME4 = ""

VECTOR_STORE_NAME = VECTOR_STORE_NAME1


# knowledge file path
KNOWLEDGE_FILE_PATH = KNOWLEDGE_FILE1
KNOWLEDGE_FILE1 = ""
KNOWLEDGE_FILE2  = ""
KNOWLEDGE_FILE3 = ""

# 本地LLM API config
ChatGLM3_6B_BASE_URL = "http://localhost:8000/v1/chat/completions"
Qwen7B_BASE_URL = "http://localhost:8001/v1/chat/completions"
Qwen14B_BASE_URL = "http://localhost:8002/v1/chat/completions"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# ragas评估数据 config
ragas_dataset_base_path = "Evaluation/data"
ragas_dataset_name = {
    "naive_rag": "naive_eval_data.json",
    "naive_test": "naive_eval_data_test.json",
    "contextual_compressed": "contextual_compressed_eval_data.json",
    "small2big": "small2big_eval_data.json",
    "rag_fusion&naive": "rag_fusion&naive_eval_data.json",
    "rag_fusion&small2big": "rag_fusion&small2big_eval_data.json",
    "rag_fusion&contextual_compression": "rag_fusion&contextual_compression_eval_data.json",

    "small2big_14": "small2big_eval_data_14.json",
    "naive_rag_14": "naive_eval_data_updated_14.json",
    "cc_14":"contextual_compressed_eval_data_14.json",
    "rf_cc_14": "rag_fusion&contextual_compression_eval_data_14.json",

    "sr_rr_nc_big": "super_retriever_real_rerank_noCOT_eval_data_14_dedu.json",
    "sr_rr_c_big": "super_retriever_real_rerank_&COT_eval_data_14_dedu.json",
    
    "rf_s2b_2ft_14" : "rag_fusion&small2big_eval_data_14_small_2ft.json",
    "rf_naive_2ft_14" : "rag_fusion&naive_eval_data_14_big_2ft.json",
    "rf_cc_2ft_14":"rag_fusion&contextual_compressed_eval_data_14_big_2ft.json",

    "cc_14_small":"contextual_compressed_eval_data_14_small.json",
    "cc_s2b_14_small": "small2big&contextual_compression_eval_data_14_small.json"

}

# 本地LLM加载 config
init_llm = "ChatGLM3-6B-32K"
bge_embedding_model = "bge-reranker-base"
text2vector_embedding_model = "text2vec-base"

# model config
embedding_model_dict = {
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "text2vec-base": "model/text2vec-large-chinese",
    'simbert-base-chinese': 'model/simbert-base-chinese',
}


llm_model_dict = {
    "ChatGLM3-6B": "model/chatglm3-6b-32k",

    "Qwen-7B":"qwen/Qwen-7B-Chat",
    "Qwen-14B":"qwen/Qwen-14B-Chat",
    "Qwen1.5-7B":"qwen/Qwen1.5-7B-Chat",
    "Qwen1.5-14B":"qwen/Qwen1.5-14B-Chat",

    "vicuna1.5-13b":"lmsys/vicuna-13b-v1.5-16k",
    "Llama-2-7b":"meta/Llama-2-7b-chat-hf"

}
