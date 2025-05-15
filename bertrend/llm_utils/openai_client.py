#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
from typing import Type, Literal
from enum import Enum

from openai import OpenAI, AzureOpenAI, Timeout, Stream
from loguru import logger
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

MAX_ATTEMPTS = 3
TIMEOUT = 60.0
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 512

AZURE_API_VERSION = "2025-03-01-preview"

# Default ports for local LLM servers
OLLAMA_DEFAULT_PORT = 11434
LM_STUDIO_DEFAULT_PORT = 1234


class EndpointType(str, Enum):
    """Enum for different types of OpenAI-compatible endpoints"""

    OPENAI = "openai"  # Official OpenAI API
    AZURE = "azure"  # Azure OpenAI service
    OLLAMA = "ollama"  # Ollama local deployment
    LM_STUDIO = "lm_studio"  # LM Studio local deployment
    OTHER = "other"  # Other OpenAI-compatible endpoints


class OpenAI_Client:
    """
    Generic client for OpenAI-compatible APIs.

    This class provides a unified interface for interacting with OpenAI models,
    supporting:
    - Official OpenAI API
    - Azure OpenAI service
    - Ollama local deployments
    - LM Studio local deployments
    - Other OpenAI-compatible endpoints

    The class handles authentication, request formatting, and error handling.

    Notes
    -----
    The API key and the ENDPOINT must be set using environment variables OPENAI_API_KEY and
    OPENAI_ENDPOINT respectively. The endpoint should be set for any non-OpenAI deployment.
    """

    def __init__(
        self,
        api_key: str = None,
        endpoint: str = None,
        model: str = None,
        temperature: float = DEFAULT_TEMPERATURE,
        api_version: str = AZURE_API_VERSION,
    ):
        """
        Initialize the OpenAI-compatible client.

        Parameters
        ----------
        api_key : str, optional
            API key. If None, will try to get from OPENAI_API_KEY environment variable.
        endpoint : str, optional
            API endpoint URL. If None, will try to get from OPENAI_ENDPOINT environment variable.
            Should be set for Azure, Ollama, LM Studio or other local deployments.
            For Ollama, use format: http://localhost:11434 or http://your-ollama-host:port
            For LM Studio, use format: http://localhost:1234 or http://your-lmstudio-host:port
        model : str, optional
            Name of the model to use. If None, will try to get from OPENAI_DEFAULT_MODEL_NAME environment variable.
        temperature : float, default=DEFAULT_TEMPERATURE
            Temperature parameter for controlling randomness in generation.
        api_version : str, default=AZURE_API_VERSION
            API version to use for Azure OpenAI service.

        Raises
        ------
        EnvironmentError
            If api_key is None and OPENAI_API_KEY environment variable is not set.
        """
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "WARNING: OPENAI_API_KEY environment variable not found. Please set it before using OpenAI services."
            )
            raise EnvironmentError(f"OPENAI_API_KEY environment variable not found.")

        endpoint = endpoint or os.getenv("OPENAI_ENDPOINT", None)
        if endpoint == "":  # check empty env var
            endpoint = None

        # Detect endpoint type
        endpoint_type = self._detect_endpoint_type(endpoint)
        logger.debug(f"Detected endpoint type: {endpoint_type}")

        # Set up common parameters for all client types
        common_params = {
            "api_key": api_key,
            "timeout": Timeout(TIMEOUT, connect=10.0),
            "max_retries": MAX_ATTEMPTS,
        }

        # Configure client based on endpoint type
        if endpoint_type == EndpointType.AZURE:
            self.llm_client = AzureOpenAI(
                **common_params,
                azure_endpoint=endpoint,
                api_version=api_version or AZURE_API_VERSION,
            )
        else:
            # For all other types (OpenAI, Ollama, LM Studio, Other), use the OpenAI client
            # with appropriate base_url
            if endpoint_type in [EndpointType.OLLAMA, EndpointType.LM_STUDIO]:
                # For local deployments, ensure the endpoint has the correct format
                endpoint = self._format_local_endpoint(endpoint, endpoint_type)

            self.llm_client = OpenAI(
                **common_params,
                base_url=endpoint,
            )

        self.model_name = model or os.getenv("OPENAI_DEFAULT_MODEL_NAME")
        self.temperature = temperature
        self.max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
        self.endpoint_type = endpoint_type

    def _detect_endpoint_type(self, endpoint: str | None) -> EndpointType:
        """
        Detect the type of endpoint based on the URL.

        Parameters
        ----------
        endpoint : str or None
            The endpoint URL to analyze.

        Returns
        -------
        EndpointType
            The detected endpoint type.
        """
        if not endpoint:
            return EndpointType.OPENAI

        endpoint = endpoint.lower()
        if "azure.com" in endpoint:
            return EndpointType.AZURE
        elif "localhost" in endpoint or "127.0.0.1" in endpoint:
            if str(OLLAMA_DEFAULT_PORT) in endpoint:
                return EndpointType.OLLAMA
            elif str(LM_STUDIO_DEFAULT_PORT) in endpoint:
                return EndpointType.LM_STUDIO
        return EndpointType.OTHER

    def _format_local_endpoint(
        self, endpoint: str | None, endpoint_type: EndpointType
    ) -> str:
        """
        Format the endpoint URL for local deployments.

        Parameters
        ----------
        endpoint : str or None
            The endpoint URL to format.
        endpoint_type : EndpointType
            The type of endpoint to format for.

        Returns
        -------
        str
            The formatted endpoint URL.
        """
        if not endpoint:
            # Use default localhost URLs if no endpoint provided
            if endpoint_type == EndpointType.OLLAMA:
                return f"http://localhost:{OLLAMA_DEFAULT_PORT}"
            elif endpoint_type == EndpointType.LM_STUDIO:
                return f"http://localhost:{LM_STUDIO_DEFAULT_PORT}"

        # Ensure the endpoint has the correct format
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"http://{endpoint}"
        return endpoint

    def generate(
        self,
        user_prompt,
        system_prompt=None,
        **kwargs,
    ) -> ChatCompletion | Stream[ChatCompletionChunk] | str:
        """
        Call OpenAI model for text generation.

        Parameters
        ----------
        user_prompt : str
            Prompt to send to the model with role=user.
        system_prompt : str, optional
            Prompt to send to the model with role=system.
        **kwargs : dict
            Additional arguments to pass to the OpenAI API.

        Returns
        -------
        str or Stream[ChatCompletionChunk]
            Model response as text, or a stream of response chunks if stream=True is passed in kwargs.
        """
        # Transform messages into OpenAI API compatible format
        messages = [{"role": "user", "content": user_prompt}]
        # Add a system prompt if one is provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return self.generate_from_history(messages, **kwargs)

    def generate_from_history(
        self,
        messages: list[dict],
        **kwargs,
    ) -> ChatCompletion | Stream[ChatCompletionChunk] | str:
        """
        Call OpenAI model for text generation using a conversation history.

        Parameters
        ----------
        messages : list[dict]
            List of message dictionaries to pass to the API in OpenAI format.
            Each message should have 'role' and 'content' keys.
        **kwargs : dict
            Additional arguments to pass to the OpenAI API.

        Returns
        -------
        str or Stream[ChatCompletionChunk]
            Model response as text, or a stream of response chunks if stream=True is passed in kwargs.
        """
        # For important parameters, set a default value if not given
        if not kwargs.get("model"):
            kwargs["model"] = self.model_name
        if not kwargs.get("temperature"):
            kwargs["temperature"] = self.temperature
        if not kwargs.get("max_output_tokens"):
            kwargs["max_output_tokens"] = self.max_output_tokens

        try:
            response = self.llm_client.responses.create(input=messages, **kwargs)
            logger.debug(f"API returned: {response}")
            if kwargs.get("stream", False):
                return response
            else:
                return response.output_text
            # Details of errors available here: https://platform.openai.com/docs/guides/error-codes/api-errors
        except Exception as e:
            msg = f"OpenAI API fatal error: {e}"
            logger.error(msg)
            return msg

    def parse(
        self,
        user_prompt: str,
        system_prompt: str = None,
        response_format: Type[BaseModel] = None,
        **kwargs,
    ) -> BaseModel | None:
        """
        Call OpenAI model for generation with structured output.

        Parameters
        ----------
        user_prompt : str
            Prompt to send to the model with role=user.
        system_prompt : str, optional
            Prompt to send to the model with role=system.
        response_format : Type[BaseModel], optional
            Pydantic model class defining the expected output structure.
        **kwargs : dict
            Additional arguments to pass to the OpenAI API.

        Returns
        -------
        BaseModel or None
            A pydantic object instance of the specified response_format type,
            or None if an error occurs.

        Notes
        -----
        This method uses the beta.chat.completions.parse API which supports
        structured outputs in the format defined by the response_format parameter.
        """
        # Transform messages into OpenAI API compatible format
        messages = [{"role": "user", "content": user_prompt}]
        # Add a system prompt if one is provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        # For important parameters, set a default value if not given
        if not kwargs.get("model"):
            kwargs["model"] = self.model_name
        if not kwargs.get("temperature"):
            kwargs["temperature"] = self.temperature

        try:
            # NB. here use beta.chat...parse to support structured outputs
            answer = self.llm_client.beta.chat.completions.parse(
                messages=messages,
                response_format=response_format,
                **kwargs,
            )
            logger.debug(f"API returned: {answer}")
            return answer.choices[0].message.parsed
            # Details of errors available here: https://platform.openai.com/docs/guides/error-codes/api-errors
        except Exception as e:
            msg = f"OpenAI API fatal error: {e}"
            logger.error(msg)
