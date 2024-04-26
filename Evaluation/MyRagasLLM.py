import typing as t
from typing import List
from ragas.llms import RagasLLM
from langchain.schema import LLMResult
from langchain.schema import Generation
from langchain.callbacks.base import Callbacks
from langchain.prompts import PromptTemplate, ChatPromptTemplate


from Chat.MyLLM import ChatGLM, Qwen7B, Qwen14B

class MyLLM_ChatGLM(RagasLLM):

    def __init__(self, llm):
        self.base_llm = llm

    @property
    def llm(self):
        return self.base_llm

    async def agenerate(
                        self,
                        prompts: list[ChatPromptTemplate],
                        n: int = 1,
                        temperature: float = 1e-8,
                        callbacks: t.Optional[Callbacks] = None
    ) -> LLMResult:

        generations = []
        llm_output = {}
        token_total = 0
        llm = ChatGLM()
        for prompt in prompts:
            content = prompt.messages[0].content
            text = self.base_llm._call(content)  # 修改为自己的API方式调用即可
            generations.append([Generation(text=text)])
            token_total += len(text)
        llm_output['token_total'] = token_total

        return LLMResult(generations=generations, llm_output=llm_output)

    def generate(
            self,
            prompts: List[PromptTemplate],
            n: int = 1,
            temperature: float = 0,
            callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        generations = []
        llm_output = {}
        token_total = 0
        # llm = ChatGLM3()
        for prompt in prompts:
            content = prompt.messages[0].content
            text = self.base_llm._call(content)  # 修改为自己的API方式调用即可
            generations.append([Generation(text=text)])
            token_total += len(text)
        llm_output['token_total'] = token_total

        return LLMResult(generations=generations, llm_output=llm_output)

class MyLLM_Qwen(RagasLLM):

    def __init__(self, llm):
        self.base_llm = llm

    @property
    def llm(self):
        return self.base_llm

    async def agenerate(
                        self,
                        prompts: list[ChatPromptTemplate],
                        n: int = 1,
                        temperature: float = 1e-8,
                        callbacks: t.Optional[Callbacks] = None
    ) -> LLMResult:

        generations = []
        llm_output = {}
        token_total = 0
        # llm = Qwen14B()
        for prompt in prompts:
            content = prompt.messages[0].content
            text = self.base_llm._call(content)  # 修改为自己的API方式调用即可
            generations.append([Generation(text=text)])
            token_total += len(text)
        llm_output['token_total'] = token_total

        return LLMResult(generations=generations, llm_output=llm_output)

    def generate(
            self,
            prompts: List[PromptTemplate],
            n: int = 1,
            temperature: float = 0,
            callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        generations = []
        llm_output = {}
        token_total = 0
        # llm = Qwen()
        for prompt in prompts:
            content = prompt.messages[0].content
            text = self.base_llm._call(content)  # 修改为自己的API方式调用即可
            generations.append([Generation(text=text)])
            token_total += len(text)
        llm_output['token_total'] = token_total

        return LLMResult(generations=generations, llm_output=llm_output)



