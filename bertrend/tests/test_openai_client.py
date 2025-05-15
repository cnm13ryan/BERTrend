#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import os
from unittest.mock import patch, MagicMock
from openai import OpenAI, AzureOpenAI, Stream
from pydantic import BaseModel

from bertrend.llm_utils.openai_client import (
    OpenAI_Client,
    EndpointType,
    OLLAMA_DEFAULT_PORT,
    LM_STUDIO_DEFAULT_PORT,
)


@pytest.fixture
def mock_api_key():
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    yield
    del os.environ["OPENAI_API_KEY"]


@pytest.fixture
def mock_azure_endpoint():
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["OPENAI_ENDPOINT"] = "https://azure.com"
    yield
    del os.environ["OPENAI_API_KEY"]
    del os.environ["OPENAI_ENDPOINT"]


def test_initialization_with_azure(mock_azure_endpoint):
    """Test client initialization when using Azure endpoint"""
    client = OpenAI_Client(api_key="test_api_key", endpoint="https://azure.com")
    assert isinstance(client.llm_client, AzureOpenAI)
    assert client.endpoint_type == EndpointType.AZURE


def test_initialization_without_azure(mock_api_key):
    """Test client initialization when using OpenAI endpoint"""
    client = OpenAI_Client(api_key="test_api_key")
    assert isinstance(client.llm_client, OpenAI)
    assert client.endpoint_type == EndpointType.OPENAI


def test_initialization_with_ollama(mock_api_key):
    """Test client initialization when using Ollama endpoint"""
    client = OpenAI_Client(
        api_key="test_key", endpoint=f"http://localhost:{OLLAMA_DEFAULT_PORT}"
    )
    assert isinstance(client.llm_client, OpenAI)
    assert client.endpoint_type == EndpointType.OLLAMA


def test_initialization_with_lm_studio(mock_api_key):
    """Test client initialization when using LM Studio endpoint"""
    client = OpenAI_Client(
        api_key="test_key", endpoint=f"http://localhost:{LM_STUDIO_DEFAULT_PORT}"
    )
    assert isinstance(client.llm_client, OpenAI)
    assert client.endpoint_type == EndpointType.LM_STUDIO


def test_endpoint_type_detection():
    """Test endpoint type detection for various URLs"""
    client = OpenAI_Client(api_key="test_key")

    # Test Azure detection
    assert client._detect_endpoint_type("https://test.azure.com") == EndpointType.AZURE
    assert client._detect_endpoint_type("http://azure.com") == EndpointType.AZURE

    # Test Ollama detection
    assert (
        client._detect_endpoint_type(f"http://localhost:{OLLAMA_DEFAULT_PORT}")
        == EndpointType.OLLAMA
    )
    assert (
        client._detect_endpoint_type(f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}")
        == EndpointType.OLLAMA
    )

    # Test LM Studio detection
    assert (
        client._detect_endpoint_type(f"http://localhost:{LM_STUDIO_DEFAULT_PORT}")
        == EndpointType.LM_STUDIO
    )
    assert (
        client._detect_endpoint_type(f"http://127.0.0.1:{LM_STUDIO_DEFAULT_PORT}")
        == EndpointType.LM_STUDIO
    )

    # Test default OpenAI
    assert client._detect_endpoint_type(None) == EndpointType.OPENAI

    # Test other endpoints
    assert (
        client._detect_endpoint_type("http://custom-endpoint:8000")
        == EndpointType.OTHER
    )


def test_local_endpoint_formatting():
    """Test endpoint formatting for local deployments"""
    client = OpenAI_Client(api_key="test_key")

    # Test Ollama formatting
    assert (
        client._format_local_endpoint(None, EndpointType.OLLAMA)
        == f"http://localhost:{OLLAMA_DEFAULT_PORT}"
    )
    assert (
        client._format_local_endpoint("custom-host:11434", EndpointType.OLLAMA)
        == "http://custom-host:11434"
    )

    # Test LM Studio formatting
    assert (
        client._format_local_endpoint(None, EndpointType.LM_STUDIO)
        == f"http://localhost:{LM_STUDIO_DEFAULT_PORT}"
    )
    assert (
        client._format_local_endpoint("custom-host:1234", EndpointType.LM_STUDIO)
        == "http://custom-host:1234"
    )

    # Test https preservation
    assert (
        client._format_local_endpoint(
            "https://custom-host:1234", EndpointType.LM_STUDIO
        )
        == "https://custom-host:1234"
    )


def test_generate_user_prompt(mock_api_key):
    """Test generation with a simple user prompt"""
    client = OpenAI_Client(api_key="test_api_key")

    # Mocking the llm_client's responses create method
    with patch.object(
        client.llm_client.responses,
        "create",
        return_value=MagicMock(output_text="This is a mock response"),
    ):
        result = client.generate("What is the weather today?")
        assert result == "This is a mock response"


def test_generate_from_history(mock_api_key):
    """Test generation using history of messages"""
    client = OpenAI_Client(api_key="test_api_key")

    # Mocking the llm_client's responses create method
    with patch.object(
        client.llm_client.responses,
        "create",
        return_value=MagicMock(output_text="This is a mock response"),
    ):
        messages = [{"role": "user", "content": "What is the weather today?"}]
        result = client.generate_from_history(messages)
        assert result == "This is a mock response"


def test_api_error_handling(mock_api_key):
    """Test if an API error is properly handled"""
    client = OpenAI_Client(api_key="test_api_key")

    # Simulate an error during API call
    with patch.object(
        client.llm_client.responses, "create", side_effect=Exception("API Error")
    ):
        result = client.generate("What is the weather today?")
        assert result == "OpenAI API fatal error: API Error"


def test_generate_with_streaming(mock_api_key):
    """Test if streaming works when 'stream' is True"""
    client = OpenAI_Client(api_key="test_api_key")

    # Mock streaming response
    mock_stream = MagicMock(spec=Stream)
    with patch.object(client.llm_client.responses, "create", return_value=mock_stream):
        result = client.generate("What is the weather today?", stream=True)
        assert result == mock_stream


# Define a test Pydantic model for parse tests
# Using underscore prefix to prevent pytest from collecting it as a test class
class _TestResponseModel(BaseModel):
    answer: str
    confidence: float


class MockMessage:
    def __init__(self, parsed):
        self.parsed = parsed


class MockChoice:
    def __init__(self, parsed):
        self.message = MockMessage(parsed)


class MockParseResponse:
    def __init__(self, parsed):
        self.choices = [MockChoice(parsed)]


def test_parse_basic_functionality(mock_api_key):
    """Test parse method with a simple user prompt"""
    client = OpenAI_Client(api_key="test_api_key")

    # Create a mock parsed response
    mock_parsed = _TestResponseModel(answer="This is a test answer", confidence=0.95)
    mock_response = MockParseResponse(mock_parsed)

    # Mock the beta.chat.completions.parse method
    with patch.object(
        client.llm_client.beta.chat.completions,
        "parse",
        return_value=mock_response,
    ):
        result = client.parse(
            "What is the weather today?", response_format=_TestResponseModel
        )
        assert isinstance(result, _TestResponseModel)
        assert result.answer == "This is a test answer"
        assert result.confidence == 0.95


def test_parse_with_system_prompt(mock_api_key):
    """Test parse method with both user and system prompts"""
    client = OpenAI_Client(api_key="test_api_key")

    # Create a mock parsed response
    mock_parsed = _TestResponseModel(answer="System prompt response", confidence=0.9)
    mock_response = MockParseResponse(mock_parsed)

    # Mock the beta.chat.completions.parse method
    with patch.object(
        client.llm_client.beta.chat.completions,
        "parse",
        return_value=mock_response,
    ) as mock_parse:
        result = client.parse(
            "What is the weather today?",
            system_prompt="You are a weather assistant",
            response_format=_TestResponseModel,
        )

        # Verify the system prompt was included in the messages
        args, kwargs = mock_parse.call_args
        messages = kwargs.get("messages", [])
        assert any(
            msg.get("role") == "system"
            and "weather assistant" in msg.get("content", "")
            for msg in messages
        )

        # Verify the result
        assert isinstance(result, _TestResponseModel)
        assert result.answer == "System prompt response"


def test_parse_error_handling(mock_api_key):
    """Test if an API error in parse is properly handled"""
    client = OpenAI_Client(api_key="test_api_key")

    # Simulate an error during API call
    with patch.object(
        client.llm_client.beta.chat.completions,
        "parse",
        side_effect=Exception("API Parse Error"),
    ):
        result = client.parse(
            "What is the weather today?", response_format=_TestResponseModel
        )
        assert result is None


def test_parse_with_none_response_format(mock_api_key):
    """Test parse method with response_format=None"""
    client = OpenAI_Client(api_key="test_api_key")

    # Create a mock parsed response
    mock_parsed = {"answer": "Default response", "confidence": 0.8}
    mock_response = MockParseResponse(mock_parsed)

    # Mock the beta.chat.completions.parse method
    with patch.object(
        client.llm_client.beta.chat.completions,
        "parse",
        return_value=mock_response,
    ) as mock_parse:
        result = client.parse("What is the weather today?", response_format=None)

        # Verify response_format was passed as None
        args, kwargs = mock_parse.call_args
        assert kwargs.get("response_format") is None

        # Verify the result
        assert result == mock_parsed
