#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
from configparser import ConfigParser

import pytest
from pathlib import Path
from unittest.mock import patch

from bertrend.utils.config_utils import (
    _resolve_env_variables,
    load_toml_config,
    parse_literal,
    EnvInterpolation,
)


@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {"VAR_NAME": "test_value", "NUMBER": "42"}):
        yield


def test_resolve_env_variables(mock_env):
    config_dict = {
        "key1": "$VAR_NAME",
        "key2": {"subkey1": "$VAR_NAME", "subkey2": "$NUMBER"},
        "key3": "no_env_var",
    }

    resolved_config = _resolve_env_variables(config_dict)

    assert resolved_config["key1"] == "test_value"
    assert resolved_config["key2"]["subkey1"] == "test_value"
    assert resolved_config["key2"]["subkey2"] == "42"
    assert resolved_config["key3"] == "no_env_var"


def test_load_toml_config(mock_env):
    toml_content = """
    [section]
    key1 = "$VAR_NAME"
    key2 = "$NUMBER"
    """

    # Create a temporary TOML file
    toml_file = Path("test_config.toml")
    toml_file.write_text(toml_content)

    try:
        config = load_toml_config(toml_file)
        assert config["section"]["key1"] == "test_value"
        assert config["section"]["key2"] == "42"
    finally:
        toml_file.unlink()  # Clean up the test file


def test_parse_literal():
    test_cases = [
        (
            {"a": "2", "b": "3", 3: "xyz", "c": "0.5", "z": "(1,1)"},
            {"a": 2, "b": 3, 3: "xyz", "c": 0.5, "z": (1, 1)},
        ),
        ('"hello"', "hello"),
        ("42", 42),
        ("(1, 2)", (1, 2)),
        ("True", True),
        ("False", False),
        ("None", None),
    ]

    for input_expr, expected_output in test_cases:
        assert parse_literal(input_expr) == expected_output


def test_env_interpolation(mock_env):
    interpolation = EnvInterpolation()

    # Mock the 'before_get' method in the EnvInterpolation class
    value = "Hello $VAR_NAME"
    result = interpolation.before_get(ConfigParser(), None, None, value, None)

    assert result == "Hello test_value"


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("3.14", 3.14),
        ("True", True),
        ("False", False),
        ("(1,2)", (1, 2)),
        ("None", None),
        ("42", 42),
    ],
)
def test_parse_literal_with_parametrize(expr, expected):
    assert parse_literal(expr) == expected
