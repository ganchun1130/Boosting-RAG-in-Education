import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (UnstructuredFileLoader,
                                        PyPDFLoader,
                                        UnstructuredPDFLoader,
                                        PDFPlumberLoader,
                                        PDFMinerLoader,
                                        PyMuPDFLoader,
                                        PyPDFium2Loader)
from chinese_text_splitter import ChineseTextSplitter

filepath = ""
# 加载PDF文件
# loader = PyMuPDFLoader(file_path=filepath)
loader = UnstructuredPDFLoader(file_path=filepath)
# splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=["\n\n", "  \n \n", "\n \n",])  # 设置一个非常大的chunk_size
splitter = ChineseTextSplitter(chunk_size=200, chunk_overlap=0, pdf=True)
docs = loader.load_and_split(splitter)

# 计算最大的chunk_size
# max_chunk_size = max(len(chunk) for chunk in docs)

print(docs)
print(len(docs))

# print(f"最大的chunk_size是：{max_chunk_size}")
