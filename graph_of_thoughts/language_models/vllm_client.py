# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import backoff
import os
import random
import time
from typing import List, Dict, Union
from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion

from .abstract_language_model import AbstractLanguageModel


class vLLMClient(AbstractLanguageModel):
    """
    The vLLMClient class handles interactions with vLLM OpenAI-compatible server using the provided configuration.
    
    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "vllm", cache: bool = False
    ) -> None:
        """
        Initialize the vLLMClient instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'vllm'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        
        # The model_id is the id of the model that is used for the vLLM server
        self.model_id: str = self.config["model_id"]
        
        # The prompt_token_cost and response_token_cost are the costs for 1000 prompt tokens and 1000 response tokens respectively.
        self.prompt_token_cost: float = self.config.get("prompt_token_cost", 0.0)
        self.response_token_cost: float = self.config.get("response_token_cost", 0.0)
        
        # The temperature of a model is defined as the randomness of the model's output.
        self.temperature: float = self.config.get("temperature", 0.7)
        
        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens: int = self.config.get("max_tokens", 512)
        
        # The stop sequence is a sequence of tokens that the model will stop generating at
        self.stop: Union[str, List[str]] = self.config.get("stop", None)
        
        # vLLM server base URL
        self.base_url: str = self.config["base_url"]
        
        # API key for vLLM server (can be dummy for local servers)
        self.api_key: str = self.config.get("api_key", "dummy-key")
        
        # Initialize the OpenAI Client pointing to vLLM server
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        Query the vLLM server for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the vLLM server.
        :rtype: Union[List[ChatCompletion], ChatCompletion]
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if num_responses == 1:
            response = self.chat([{"role": "user", "content": query}], num_responses)
        else:
            response = []
            next_try = num_responses
            total_num_attempts = num_responses
            while num_responses > 0 and total_num_attempts > 0:
                try:
                    assert next_try > 0
                    res = self.chat([{"role": "user", "content": query}], next_try)
                    response.append(res)
                    num_responses -= next_try
                    next_try = min(num_responses, next_try)
                except Exception as e:
                    next_try = (next_try + 1) // 2
                    self.logger.warning(
                        f"Error in vLLM client: {e}, trying again with {next_try} samples"
                    )
                    time.sleep(random.randint(1, 3))
                    total_num_attempts -= 1

        if self.cache:
            self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict], num_responses: int = 1) -> ChatCompletion:
        """
        Send chat messages to the vLLM server and retrieves the model's response.
        Implements backoff on OpenAI error.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The vLLM server's response.
        :rtype: ChatCompletion
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=num_responses,
            stop=self.stop,
        )

        # Update token counts if available
        if hasattr(response, 'usage') and response.usage:
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            prompt_tokens_k = float(self.prompt_tokens) / 1000.0
            completion_tokens_k = float(self.completion_tokens) / 1000.0
            self.cost = (
                self.prompt_token_cost * prompt_tokens_k
                + self.response_token_cost * completion_tokens_k
            )
        
        self.logger.info(
            f"Response from vLLM server: {response}"
            f"\nCost estimate: {self.cost}"
        )
        return response

    def get_response_texts(
        self, query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        Extract the response texts from the vLLM server's response.

        :param query_response: The response(s) from the query method.
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: List of response strings.
        :rtype: List[str]
        """
        if isinstance(query_response, list):
            return [
                choice.message.content
                for response in query_response
                for choice in response.choices
            ]
        else:
            return [choice.message.content for choice in query_response.choices]