import os
import re
from pathlib import Path
from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import TextLoader
from openai import OpenAI as openAI

from langchain.chains import RetrievalQA, LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.document_transformers import EmbeddingsRedundantFilter

from llama_hub.file.pdf.base import PDFReader
from llama_index.response.notebook_utils import display_source_node
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import IndexNode
from llama_index.embeddings import resolve_embed_model
from pydantic import BaseModel, Field

from RAG.retrieval.retriever import small2big_contextual_compression_retriever
from config import *
from chinese_text_splitter import ChineseTextSplitter
from RAG.retrieval.rerank import BgeRerank
from RAG.retrieval.retriever import *


# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


prompt_template = """你是一个真诚且友善的智能助手，你现在服务于一个名为小雅的智能教育系统。
                            现在有用户向你询问系统的某些使用方法，请你基于以下已知内容来清楚地、有条理地以及精确地回答用户的问题。你应该重点关注排序靠前的已知内容。
                            如果你无法从中得到答案，请说没有提供足够的相关信息。不允许在答案中添加编造成分。另外，答案请使用中文。
                            不要回答与用户问题无关的内容。

                            用户问题：{question},已知内容：{context}。
                            """

prompt = PromptTemplate(template=prompt_template,
                        input_variables=["context", "question"])


def remove_leading_numbers_from_queries(queries):
    cleaned_queries = []
    for query in queries:
        cleaned_query = re.sub(r'^\d+\.\s*', '', query)
        cleaned_queries.append(cleaned_query)
    return cleaned_queries


# Function to generate queries using OpenAI's ChatGPT
def rag_fusion(original_query):
    client = openAI(api_key="EMPTY", base_url=ChatGLM_BASE_URL)
    messages = [
        {"role": "system",
         "content": "你是一个真诚且友善的智能助手，你现在服务于一个名为小雅的智能教育系统。你十分了解这个系统的所有功能。"
         },
        {"role": "user",
         "content": f"请根据用户的问题 “{original_query}”，可以从多个视角生成与之相似的问题，以便于更全面地检索和回答用户可能关心的问题。"
                    f"这些问题可以涵盖不同方面。生成的每个问题都应具有较高的相关性和实用性。您的目标是帮助用户克服基于距离的相似性搜索的一些限制。"},
        {"role": "user",
         "content": "请生成用换行符分隔的4~5个相似问题。"}
    ]
    response = client.chat.completions.create(
        model="ChatGLM3-6B-32K",
        messages=messages,
    )
    if response:
        content = response.choices[0].message.content.strip().split("\n")
        cleaned_content = remove_leading_numbers_from_queries(content)
        print("生成的相似问题如下：")
        print(cleaned_content)
        return cleaned_content
    else:
        print("Error:", response.status_code)


def naive_rag(llm, embeddings, top_k, rerank: bool):
    vector_store = FAISS.load_local(os.path.join(VECTOR_STORE_PATH, VECTOR_STORE_NAME), embeddings)
    retriever = vector_store.as_retriever(
        search_kwargs={"k": top_k})

    if rerank:
        reranker = BgeRerank(top_n=top_k)
        naive_retriever = ContextualCompressionRetriever(base_compressor=reranker,
                                                         base_retriever=retriever,
                                                         search_kwargs={"k": top_k})
    else:
        naive_retriever = retriever

    knowledge_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=naive_retriever,
        prompt=prompt)
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}")
    knowledge_chain.return_source_documents = True

    return knowledge_chain


def multi_query_langchain(llm1, llm2, embeddings, top_k, retriever_type, rerank: bool, file_path=None):
    """
        https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever
        """
    vector_store = FAISS.load_local(os.path.join(VECTOR_STORE_PATH, VECTOR_STORE_NAME), embeddings)
    output_parser = LineListOutputParser()

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""你是一个真诚且友善的智能助手，你现在服务于一个名为小雅的智能教育系统。你十分了解这个系统的所有功能。
                    请根据用户的问题: “{question}”，首先对用户问题的所属模块以及问答类型进行分类，然后从多个视角生成与之相似的问题，以便于更全面地检索和回答用户可能关心的问题。
                    这些问题可以涵盖不同方面。生成的每个问题都应具有较高的相关性和实用性。您的目标是帮助用户克服基于距离的相似性搜索的一些限制。
                    请在第一点表明用户问题的所属模块以及问答类型，然后生成用换行符分隔的4个相似问题。
                    这是一个样例：
                    query:请问小雅可以签到吗
                    answer:
                    1. 所属模块:'签到'，问答类型:'功能不了解'。
                    2. 小雅平台如何签到？
                    3. 签到功能在哪里可以找到？
                    4. 如何使用小雅平台的签到功能？
                    5. 在小雅平台上如何设置签到提醒？
                    """,
    )
    # Chain
    llm_chain = LLMChain(llm=llm1, prompt=QUERY_PROMPT, output_parser=output_parser)
    # set retriever type
    if "all" in retriever_type.lower():
        print(f"you are using super retriever function to build your eval dataset!")
        # 加载PDF文件
        pure_retriever = small2big_contextual_compression_retriever(vector_store=vector_store,
                                                                    embeddings=embeddings,
                                                                    top_k=top_k,
                                                                    file_path=file_path,
                                                                    rerank=rerank,
                                                                    similarity_threshold=0.5
                                                                    )

    elif "small2big" in retriever_type.lower():

        print(f"you are using rag_fusion&small2big function to build your eval dataset!")
        # 加载PDF文件
        loader = PyMuPDFLoader(file_path=file_path)
        docs = loader.load()
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                         chunk_overlap=0,
                                                         separators=["  \n \n", "\n\n", "\n \n"])  # 设置一个非常大的chunk_size
        child_splitter = ChineseTextSplitter(chunk_size=128,
                                             chunk_overlap=0)
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": top_k},
        )
        retriever.add_documents(docs, ids=None)

        if rerank:
            reranker = BgeRerank(top_n=top_k)  # 定义一个重排序
            pure_retriever = ContextualCompressionRetriever(base_compressor=reranker,
                                                            base_retriever=retriever)
        else:
            pure_retriever = retriever


    elif "contextual_compression" in retriever_type.lower():

        print(f"you are using rag_fusion&contextual_compression function to build your eval dataset!")

        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        splitter = ChineseTextSplitter(chunk_size=128, chunk_overlap=0)
        # splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)

        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.5)

        if rerank:
            reranker = BgeRerank(top_n=top_k)  # 定义一个重排序
            pipeline_compressor = DocumentCompressorPipeline(
                transformers=[reranker, splitter, redundant_filter, relevant_filter]
            )
        else:
            pipeline_compressor = DocumentCompressorPipeline(
                transformers=[splitter, redundant_filter, relevant_filter]
            )

        pure_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )

    else:
        print(f"you are using rag_fusion&naive function to build your eval dataset!")
        pure_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    retriever = MultiQueryRetriever(
        retriever=pure_retriever, llm_chain=llm_chain, parser_key="lines", search_kwargs={"k": top_k},
    )

    # similiar_queries = retriever.generate_queries("作业发布后还可以再编辑吗？")

    # test = retriever.get_relevant_documents("作业发布后还可以再编辑吗？")
    # print(test)

    knowledge_chain = RetrievalQA.from_llm(
        llm=llm2,
        retriever=retriever,
        prompt=prompt,
    )
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )
    knowledge_chain.return_source_documents = True

    return knowledge_chain


def small2big_llama_index(knowledge_file_path, llm, embeddings, top_k):
    # 我们使用PDFReader加载PDF文件，并将文档的每一页合并为一个document对象。
    loader = PDFReader()
    docs0 = loader.load_data(file=Path(knowledge_file_path))

    doc_text = "\n\n".join([d.get_content() for d in docs0])
    docs = [Document(text=doc_text)]

    # 将文档解析为文本块(节点)
    node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=0)
    base_nodes = node_parser.get_nodes_from_documents(docs)

    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embeddings
    )

    sub_chunk_sizes = [128, 256, 512]
    sub_node_parsers = [
        SimpleNodeParser.from_defaults(chunk_size=c, chunk_overlap=0) for c in sub_chunk_sizes
    ]

    all_nodes = []
    for base_node in base_nodes:
        for n in sub_node_parsers:
            sub_nodes = n.get_nodes_from_documents([base_node])
            sub_inodes = [
                IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
            ]
            all_nodes.extend(sub_inodes)

        # also add original node to node
        original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        all_nodes.append(original_node)

    all_nodes_dict = {n.node_id: n for n in all_nodes}

    vector_index_chunk = VectorStoreIndex(
        all_nodes, service_context=service_context
    )
    vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=top_k)
    retriever_chunk = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_chunk},
        node_dict=all_nodes_dict,
        verbose=True,
    )
    query_engine_chunk = RetrieverQueryEngine.from_args(retriever_chunk, service_context=service_context)

    return query_engine_chunk, retriever_chunk


def small2big_langchain(llm, embeddings, top_k, file_path, rerank: bool):
    # 详情请见 https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever

    # 加载PDF文件
    loader = PyMuPDFLoader(file_path=file_path)
    docs = loader.load()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0,
                                                     separators=["  \n \n", "\n\n", "\n \n",
                                                                 ])  # 设置一个非常大的chunk_size
    child_splitter = ChineseTextSplitter(chunk_size=128, chunk_overlap=0)
    vector_store = FAISS.load_local(os.path.join(VECTOR_STORE_PATH, VECTOR_STORE_NAME), embeddings)

    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": top_k}
    )
    retriever.add_documents(docs, ids=None)

    if rerank:
        reranker = BgeRerank(top_n=top_k)  # 定义一个重排序
        pure_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=retriever
        )
    else:
        pure_retriever = retriever

    knowledge_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=pure_retriever,
        prompt=prompt,
    )
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}")
    knowledge_chain.return_source_documents = True

    return knowledge_chain


def contextual_compression(llm, embeddings, top_k, rerank: bool, retriever_type, similarity_threshold=0.5, file_path=None):
    # 参考 ：https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression

    if "small2big" in retriever_type.lower():
        print("s2bcc")
        retriever = small2big_retriever(vector_store=FAISS.load_local(os.path.join(VECTOR_STORE_PATH, VECTOR_STORE_NAME), embeddings),
                                        top_k=top_k,
                                        file_path=file_path,
                                        rerank=False)
    else:
        print("cc")
        retriever = FAISS.load_local(os.path.join(VECTOR_STORE_PATH, VECTOR_STORE_NAME), embeddings).as_retriever(
            search_kwargs={"k": top_k})

    splitter = ChineseTextSplitter(chunk_size=128, chunk_overlap=0)

    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=similarity_threshold)

    if rerank:
        reranker = BgeRerank(top_n=top_k)  # 定义一个重排序
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[reranker, splitter, redundant_filter, relevant_filter]
        )
    else:
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )

    knowledge_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=compression_retriever,
        prompt=prompt)
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}")
    knowledge_chain.return_source_documents = True

    return knowledge_chain


def Boost_RAG_with_TCoT_brage(llm, retriever, query):
    print("\n\n> Question:")
    print(query)

    retrieved_info_list = retriever.get_relevant_documents(query)
    retrieved_info = []
    print("\n> source document：")
    print(f"一共有{len(retrieved_info_list)}条相关文档！")
    for doc in retrieved_info_list:
        print(doc.page_content)
        retrieved_info.append(doc.page_content)

    # 生成初步的链式思维（CoT）
    prompt_once = \
        f"""这是用户的问题：{query} ；
            这是检索到的内容：{retrieved_info}；

            请仔细阅读上述用户问题和检索到的内容。您现在有两个任务：
            第一：如果检索到的信息能够回答用户的问题，即便检索到的内容不全面，请发挥你的逻辑推理能力，一步一步地思考并构建具有结构化的答案；
            第二：如果检索到的信息实在不能回答用户的问题，或者存在非常大的不确定性，请在答案开头明确表示“需要更多信息：”，然后说明还要继续检索哪些信息。
            例如需要二次检索时，下面是一个案例：
                “user”：“作业发布后还可以在编辑啊吗？”
                “context”：“建立作业/测练资源后，可在其中添加单选题、多选题、填空题、判断题、简答题、附件题、投票、题组、编程题、量表题、排序题和匹配题12种题型。在编辑状态下点击某一个题型，即可在页面呈现该题模板，可以录入题目内容、设置正确答案、答案解析、标注知识点等。
                ”answer“：”需要更多信息：请提供更多关于作业发布后的信息。“     
        """
    ans = llm._call(prompt_once)
    # Print the result

    print("\n> Answer:")
    print(ans)

    # 判断是否需要继续检索
    if "需要更多信息" in ans:
        print("需要进行二次检索！")
        # 提取需要检索的query
        match = re.search(r'需要更多信息：(.*)', str(ans))

        if match:
            new_query = match.group(1)
            print(new_query)
        else:
            new_query = ans
            print(new_query)

        # 进行第二次检索
        additional_info = []
        docs2 = retriever.get_relevant_documents(new_query)
        for doc2 in docs2:
            additional_info.append(doc2.page_content)
        retrieved_info.extend(additional_info)

        # 去重
        retrieved_info_de_duplicate = []
        for item in retrieved_info:
            if item not in retrieved_info_de_duplicate:
                retrieved_info_de_duplicate.append(item)

        prompt_twice = \
            f"""这是用户的问题：{query} ；
                这是检索到的内容：{retrieved_info_de_duplicate}；

                请仔细阅读上述用户问题和检索到的内容。请发挥你的逻辑推理能力，一步一步地思考并构建具有结构化的答案。
                如果你发现这仍不能回答用户的问题，请表示“我的知识库中并未收集此类问题，因此我还不能回答这个问题。”              
            """

        # 生成修正后的链式思维（CoT）
        ans = llm._call(prompt_twice)
        print("\n>Twice Answer:")
        print(ans)

    return ans, retrieved_info


# Helper function for printing docs
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
