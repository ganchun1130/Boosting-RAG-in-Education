import logging
from typing import List
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader, PyMuPDFLoader, UnstructuredPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from chinese_text_splitter import ChineseTextSplitter
from config import *

embeddings1 = HuggingFaceBgeEmbeddings(model_name=embedding_model_dict[init_embedding_model])
embeddings2 = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model])

def load_file_with_sentence(filepath):

    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(filepath)
        textsplitter = RecursiveCharacterTextSplitter(pdf=True, chunk_size=500, chunk_overlap=0)
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, chunk_size=500, chunk_overlap=0)
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs

def load_file_with_chunk3(filepath):

    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(filepath)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=["  \n \n", "\n\n", "\n \n",
                                                                                       ])  # 设置一个非常大的chunk_size
        # loader = UnstructuredPDFLoader(file_path=filepath)
        # text_splitter = ChineseTextSplitter(chunk_size=200, chunk_overlap=0, pdf=True)

        docs = loader.load_and_split(text_splitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=0, separators=["\n\n"])
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs


def init_knowledge_vector_store(filepath: str or List[str], ):

    if not os.path.exists(filepath):
        return "路径不存在"
    elif os.path.isfile(filepath):
        file = os.path.split(filepath)[-1]
        try:
            docs = load_file_with_chunk3(filepath)
            print(f"{file} 已成功加载")

        except Exception as e:
            print(e)
            print(f"{file} 未能成功加载")
            return f"{file} 未能成功加载"
    elif os.path.isdir(filepath):
        docs = []
        for file in os.listdir(filepath):
            fullfilepath = os.path.join(filepath, file)
            try:
                docs += load_file_with_chunk3(fullfilepath)
                print(f"{file} 已成功加载")

            except Exception as e:
                print(e)
                print(f"{file} 未能成功加载")

    print(len(docs))
    if len(docs) > 0:
            vector_store = FAISS.from_documents(
                docs,
                embeddings2,
            )
            # vector_store.add_documents(docs)
            vector_store.save_local(os.path.join(VECTOR_STORE_PATH, "XiaoYa_faiss_with_big_chunk_questions_text2vect"))
            print("向量知识库已经成功创建！")
    else:
        print("文件均未成功加载，请检查依赖包或文件路径。")

if  __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    logging.info(f"您好！您现在正在使用{init_embedding_model}构建向量知识库！您应该使用模型{init_llm}")
    init_knowledge_vector_store(os.path.join(KNOWLEDGE_FILE_PATH, KNOWLEDGE_FILE2))
