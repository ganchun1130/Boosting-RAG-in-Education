import json
import logging
from typing import Any, List, Mapping, Optional

import requests
from langchain.llms import chatglm, tongyi
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_community.llms.utils import enforce_stop_tokens

from config import ChatGLM3_6B_BASE_URL, Qwen7B_BASE_URL, Qwen14B_BASE_URL
logger = logging.getLogger(__name__)


class ChatGLM(LLM):
    """ChatGLM3 LLM service.

    Example:
        .. code-block:: python

            from langchain_community.llms import ChatGLM3
            endpoint_url = (
                "http://127.0.0.1:8000"
            )
            ChatGLM_llm = ChatGLM3(
                endpoint_url=endpoint_url
            )
    """

    endpoint_url: str = ChatGLM3_6B_BASE_URL
    """Endpoint URL to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""
    max_token: int = 20000
    """Max token allowed to pass to the model."""
    temperature: float = 0.1
    """LLM model temperature from 0 to 10."""
    history: List[List] = []
    """History of the conversation"""
    top_p: float = 0.7
    """Top P for nucleus sampling from 0 to 1"""
    with_history: bool = False
    """Whether to use history or not"""

    @property
    def _llm_type(self) -> str:
        return "chat_glm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_url": self.endpoint_url},
            **{"model_kwargs": _model_kwargs},
        }

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Call out to a ChatGLM3 LLM inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = chatglm_llm("Who are you?")
        """

        _model_kwargs = self.model_kwargs or {}

        # HTTP headers for authorization
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-c4GOPTvMpPQojDDL95F89aE0F8A344F7A2514d943bDe2714"  # replace $OPENAI_API_KEY with your actual key
        }

        payload = {
            "model": "ChatGLM3-6B-32K",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        payload.update(_model_kwargs)
        payload.update(kwargs)

        logger.debug(f"ChatGLM3 payload: {payload}")

        # call api
        try:
            response = requests.post(url=self.endpoint_url, headers=headers, json=payload)
            # print(response)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        logger.debug(f"ChatGLM3 response: {response}")

        if response.status_code != 200:
            raise ValueError(f"Failed with response: {response}")

        try:
            parsed_response = response.json()

            # Check if response content does exists
            if isinstance(parsed_response, dict):

                # 将JSON字符串解析为Python字典
                # data = json.loads(parsed_response)
                # 提取content内容
                text = parsed_response['choices'][0]['message']['content']
                # print(text)
                # content_keys = "response"
                # if content_keys in parsed_response:
                #     text = parsed_response[content_keys]
                # else:
                #     raise ValueError(f"No content in response : {parsed_response}")
            else:
                raise ValueError(f"Unexpected response type: {parsed_response}")

        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised during decoding response from inference endpoint: {e}."
                f"\nResponse: {response.text}"
            )

        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        if self.with_history:
            self.history = self.history + [[None, parsed_response["response"]]]
        return text

class Qwen7B(LLM):
    """ChatGLM3 LLM service.

    Example:
        .. code-block:: python

            from langchain_community.llms import ChatGLM3
            endpoint_url = (
                "http://127.0.0.1:8000"
            )
            ChatGLM_llm = ChatGLM3(
                endpoint_url=endpoint_url
            )
    """

    endpoint_url: str = Qwen7B_BASE_URL
    """Endpoint URL to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""
    max_token: int = 20000
    """Max token allowed to pass to the model."""
    temperature: float = 0.3
    """LLM model temperature from 0 to 10."""
    history: List[List] = []
    """History of the conversation"""
    top_p: float = 0.7
    """Top P for nucleus sampling from 0 to 1"""
    with_history: bool = False
    """Whether to use history or not"""

    @property
    def _llm_type(self) -> str:
        return "qwen"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_url": self.endpoint_url},
            **{"model_kwargs": _model_kwargs},
        }

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Call out to a ChatGLM3 LLM inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = chatglm_llm("Who are you?")
        """

        _model_kwargs = self.model_kwargs or {}

        # HTTP headers for authorization
        headers = {
            "Content-Type": "application/json",
            # "Authorization": "Bearer sk-c4GOPTvMpPQojDDL95F89aE0F8A344F7A2514d943bDe2714"  # replace $OPENAI_API_KEY with your actual key
        }

        payload = {
            "model": "qwen",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        payload.update(_model_kwargs)
        payload.update(kwargs)

        logger.debug(f"qwen payload: {payload}")

        # call api
        try:
            response = requests.post(url=self.endpoint_url,
                                     headers=headers,
                                     json=payload
                                     )
            # print(response)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        logger.debug(f"qwen response: {response}")

        if response.status_code != 200:
            raise ValueError(f"Failed with response: {response}")

        try:
            parsed_response = response.json()

            # Check if response content does exists
            if isinstance(parsed_response, dict):

                # 将JSON字符串解析为Python字典
                # data = json.loads(parsed_response)
                # 提取content内容
                text = parsed_response['choices'][0]['message']['content']
                # print(text)
                # content_keys = "response"
                # if content_keys in parsed_response:
                #     text = parsed_response[content_keys]
                # else:
                #     raise ValueError(f"No content in response : {parsed_response}")
            else:
                raise ValueError(f"Unexpected response type: {parsed_response}")

        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised during decoding response from inference endpoint: {e}."
                f"\nResponse: {response.text}"
            )

        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        if self.with_history:
            self.history = self.history + [[None, parsed_response["response"]]]
        return text

class Qwen14B(LLM):
    """ChatGLM3 LLM service.

    Example:
        .. code-block:: python

            from langchain_community.llms import ChatGLM3
            endpoint_url = (
                "http://127.0.0.1:8000"
            )
            ChatGLM_llm = ChatGLM3(
                endpoint_url=endpoint_url
            )
    """

    endpoint_url: str = Qwen14B_BASE_URL
    """Endpoint URL to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""
    max_token: int = 20000
    """Max token allowed to pass to the model."""
    temperature: float = 0.3
    """LLM model temperature from 0 to 10."""
    history: List[List] = []
    """History of the conversation"""
    top_p: float = 0.7
    """Top P for nucleus sampling from 0 to 1"""
    with_history: bool = False
    """Whether to use history or not"""

    @property
    def _llm_type(self) -> str:
        return "qwen"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_url": self.endpoint_url},
            **{"model_kwargs": _model_kwargs},
        }

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Call out to a ChatGLM3 LLM inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = chatglm_llm("Who are you?")
        """

        _model_kwargs = self.model_kwargs or {}

        # HTTP headers for authorization
        headers = {
            "Content-Type": "application/json",
            # "Authorization": "Bearer sk-c4GOPTvMpPQojDDL95F89aE0F8A344F7A2514d943bDe2714"  # replace $OPENAI_API_KEY with your actual key
        }

        payload = {
            "model": "qwen",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        payload.update(_model_kwargs)
        payload.update(kwargs)

        logger.debug(f"qwen payload: {payload}")

        # call api
        try:
            response = requests.post(url=self.endpoint_url,
                                     headers=headers,
                                     json=payload
                                     )
            # print(response)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        logger.debug(f"qwen response: {response}")

        if response.status_code != 200:
            raise ValueError(f"Failed with response: {response}")

        try:
            parsed_response = response.json()

            # Check if response content does exists
            if isinstance(parsed_response, dict):

                # 将JSON字符串解析为Python字典
                # data = json.loads(parsed_response)
                # 提取content内容
                text = parsed_response['choices'][0]['message']['content']
                # print(text)
                # content_keys = "response"
                # if content_keys in parsed_response:
                #     text = parsed_response[content_keys]
                # else:
                #     raise ValueError(f"No content in response : {parsed_response}")
            else:
                raise ValueError(f"Unexpected response type: {parsed_response}")

        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised during decoding response from inference endpoint: {e}."
                f"\nResponse: {response.text}"
            )

        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        if self.with_history:
            self.history = self.history + [[None, parsed_response["response"]]]
        return text
