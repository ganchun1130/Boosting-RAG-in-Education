import os

from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain.retrievers.multi_query import LineListOutputParser,MultiQueryRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS


from chinese_text_splitter import ChineseTextSplitter
from config import *
from RAG.retrieval.rerank import BgeRerank


def naive_retriever(vector_store, top_k, rerank: bool):
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    if rerank:
        reranker = BgeRerank(top_n=top_k)
        compress_retriever = ContextualCompressionRetriever(base_compressor=reranker,
                                                            base_retriever=retriever,
                                                            search_kwargs={"k": top_k})
        return compress_retriever
    else:
        return retriever


def small2big_retriever(vector_store, top_k, file_path, rerank: bool):
    loader = PyMuPDFLoader(file_path=file_path)
    docs = loader.load()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=["  \n \n", "\n\n", "\n \n"])
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0, separators=["。", "！"])
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": top_k},
    )

    retriever.add_documents(documents=docs, ids=None)

    if rerank:
        reranker = BgeRerank(top_n=top_k)
        compress_retriever = ContextualCompressionRetriever(base_compressor=reranker,
                                                            base_retriever=retriever,
                                                            search_kwargs={"k": top_k})
        return compress_retriever
    else:
        return retriever


def contextual_compression_retriever(vector_store, embeddings, top_k, rerank: bool):
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)

    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.5)

    if rerank:
        reranker = BgeRerank(top_n=top_k)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[reranker, splitter, redundant_filter, relevant_filter]
        )
    else:
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever,search_kwargs={"k": top_k},
    )
    return compression_retriever


def multi_query_retriever(llm, retrievers, top_k, rerank: bool):

    output_parser = LineListOutputParser()

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""你是一个真诚且友善的智能助手，你现在服务于一个名为小雅的智能教育系统。你十分了解这个系统的所有功能。
                        请根据用户的问题，可以从多个视角生成与之相似的问题，以便于更全面地检索和回答用户可能关心的问题。
                        这些问题可以涵盖不同方面。生成的每个问题都应具有较高的相关性和实用性。
                        您的目标是帮助用户克服基于距离的相似性搜索的一些限制。
                        请生成用换行符分隔的4~5个相似问题。
                        """,
    )
    # Chain
    llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)
    # set retriever type
    retriever = MultiQueryRetriever(
        retriever=retrievers,
        llm_chain=llm_chain,
        parser_key="lines",
        search_kwargs={"k": top_k},
    )  # "lines" is the key (attribute name) of the parsed output
    if rerank:
        reranker = BgeRerank(top_n=top_k)
        compress_retriever = ContextualCompressionRetriever(base_compressor=reranker,
                                                            base_retriever=retriever,
                                                            search_kwargs={"k": top_k})
        return compress_retriever
    else:
        return retriever


def small2big_contextual_compression_retriever(vector_store, embeddings, top_k, file_path, rerank: bool, similarity_threshold):
    retriever = small2big_retriever(vector_store, top_k=top_k, file_path=file_path, rerank=False)
    # retriever = vector_store.as_retriever(top_k=top_k)
    # splitter = ChineseTextSplitter(chunk_size=200, chunk_overlap=0, pdf=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=["。", "！"])

    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=similarity_threshold)

    if rerank:
        reranker = BgeRerank(top_n=top_k)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[reranker, splitter, redundant_filter, relevant_filter]
        )
    else:
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever, search_kwargs={"k": top_k},
    )
    return compression_retriever


def super_retriever_msc(vector_store, llm1, embeddings, top_k, rerank:bool, similarity_threshold, file_path=None):
    retriever = small2big_contextual_compression_retriever(vector_store=vector_store,
                                                           embeddings=embeddings,
                                                           top_k=top_k,
                                                           rerank=True,
                                                           similarity_threshold=similarity_threshold,
                                                           file_path=file_path)
    output_parser = LineListOutputParser()

    QUERY_PROMPT1 = PromptTemplate(
        input_variables=["question"],
        template="""你是一个真诚且友善的智能助手，你现在服务于一个名为小雅的智能教育系统。你十分了解这个系统的所有功能。
                            请根据用户的问题: {question}，可以从多个视角生成与之相似的问题，以便于更全面地检索和回答用户可能关心的问题。
                            这些问题可以涵盖不同方面。生成的每个问题都应具有较高的相关性和实用性。您的目标是帮助用户克服基于距离的相似性搜索的一些限制。
                            请生成用换行符分隔的4个相似问题。
                            """,
    )
    QUERY_PROMPT2 = PromptTemplate(
        input_variables=["question"],
        template="""你是一个真诚且友善的智能助手，你现在服务于一个名为小雅的智能教育系统。
                    请根据用户的问题: {question}：
                    首先你需要判断用户的问题属于哪一类模块以及问题的性质；
                    其次可以从多个视角生成与之相似的问题，以便于更全面地检索和回答用户可能关心的问题。
                    这些问题可以涵盖不同方面。生成的每个问题都应具有较高的相关性和实用性。您的目标是帮助用户克服基于距离的相似性搜索的一些限制。
                    注意：请在第一点中回答用户问题属于哪一类模块以及问题的性质，然后请生成用换行符分隔的4个相似问题。
                """,
    )
    # Chain
    llm_chain = LLMChain(llm=llm1, prompt=QUERY_PROMPT1, output_parser=output_parser)
    pure_retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=llm_chain, parser_key="lines", search_kwargs={"k": top_k},
    )
    if rerank:
        reranker = BgeRerank(top_n=top_k)
        rerank_retriever = ContextualCompressionRetriever(base_compressor=reranker,
                                                          base_retriever=pure_retriever,
                                                          search_kwargs={"k": top_k}
                                                          )
        return rerank_retriever
    else:
        return pure_retriever

if __name__ == '__main__':
    embeddings1 = HuggingFaceBgeEmbeddings(model_name=embedding_model_dict[bge_embedding_model])
    embeddings2 = HuggingFaceEmbeddings(model_name=embedding_model_dict[text2vector_embedding_model])
    # vector_store = Chroma(collection_name="full_documents", embedding_function=embeddings)
    vector_store = FAISS.load_local(os.path.join(VECTOR_STORE_PATH, VECTOR_STORE_NAME1), embeddings=embeddings1)
    retriever = small2big_contextual_compression_retriever(embeddings=embeddings1,
                                                           vector_store=vector_store,
                                                           top_k=10,
                                                           file_path="/usr/local/TFBOYS/gc/NLP/ChatXIaoYa_chatglm/knowledge_file/promax_data_zh_processed2.pdf",
                                                           rerank=True,
                                                           similarity_threshold=0.5
                                                           )
    # retriever = contextual_compression_retriever(vector_store=vector_store, embeddings=embeddings1, top_k=10)
    # retriever = small2big_retriever(vector_store=vector_store, top_k=10, file_path="/usr/local/TFBOYS/gc/NLP/ChatXIaoYa_chatglm/knowledge_file/promax_data_zh_processed2.pdf")
    docs = retriever.get_relevant_documents(query="作业？")
    print(len(docs))
    for doc in docs:
        print(doc.page_content)