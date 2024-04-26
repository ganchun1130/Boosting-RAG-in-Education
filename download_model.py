# #模型下载
# from modelscope import snapshot_download
#
# model_dir = snapshot_download('qwen/Qwen-7B-Chat', cache_dir="/usr/local/TFBOYS/gc/model/")
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Xorbits/bge-reranker-base', cache_dir="/usr/local/TFBOYS/gc/model/")
