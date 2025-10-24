"""
ShellScriptBuilder test suite
"""

import os
import pytest
import logging
import shutil
from unittest.mock import MagicMock, patch

from pipeline import ShellScriptBuilder

@pytest.fixture(params=[
    ["datasource1"],
    ["datasource1", "datasource2"],
])
def datasources(request: pytest.FixtureRequest) -> list:
    """
    Fixture to parameterize test cases with different sets of datasources.

    Args:
        request (pytest.FixtureRequest): The pytest request object.

    Returns:
        list: A list of datasources to be used in a test case.
    """
    return request.param

@pytest.fixture(params=[
    [],
    ["arg1"],
    ["arg1", "arg2", "arg3"],
])
def args(request: pytest.FixtureRequest) -> list:
    """
    Fixture to parameterize test cases with different sets of arguments.

    Args:
        request (pytest.FixtureRequest): The pytest request object.

    Returns:
        list: A list of arguments to be used in a test case.
    """
    return request.param

@pytest.fixture(params=[
    [],
    ["echo 'Setup'"],
    ["echo 'Setup 1'", "echo 'Setup 2'"],
])
def pre_commands(request: pytest.FixtureRequest) -> list:
    """
    Fixture to parameterize test cases with different pre-commands.

    Args:
        request (pytest.FixtureRequest): The pytest request object.

    Returns:
        list: A list of pre-commands to be used in a test case.
    """
    return request.param

@pytest.fixture
def base_config() -> dict:
    """
    Fixture to provide a base configuration dictionary for the tests.

    Returns:
        dict: The base configuration dictionary.
    """
    return {
        "base_path": "/mock/base/path",
        "scripts_path": "mock/scripts",
        "datasources": {
            "datasource1": {"file_name": "datafile1.txt"},
            "datasource2": {"file_name": "datafile2.txt"}
        }
    }

@pytest.fixture
def mock_configuration(base_config: dict, datasources: list, args: list, pre_commands: list) -> dict:
    """
    Fixture to generate a mock configuration by combining base configuration and parameters.

    Args:
        base_config (dict): The base configuration dictionary.
        datasources (list): The datasources to include in the configuration.
        args (list): The arguments to include in the configuration.
        pre_commands (list): The pre-commands to include in the configuration.

    Returns:
        dict: A complete mock configuration dictionary.
    """
    test_case_name = f"test_case_{args}_{datasources}_{pre_commands}"
    config = {
        "datasources": datasources,
        "commands": ["echo 'Running test with given parameters'"],
    }

    if args:
        config["args"] = args
    if pre_commands:
        config["pre_commands"] = pre_commands

    base_config["test_cases"] = {test_case_name: config}
    return base_config

@pytest.fixture
def mock_logger() -> MagicMock:
    """
    Fixture to provide a mock logger for the tests.

    Returns:
        MagicMock: A mock logger instance.
    """
    return logging.getLogger("test_logger")

@pytest.fixture
def shell_script_builder(mock_configuration: dict, mock_logger: MagicMock) -> ShellScriptBuilder:
    """
    Fixture to provide a ShellScriptBuilder instance with mocked dependencies.

    Args:
        mock_configuration (dict): The mock configuration dictionary.
        mock_logger (MagicMock): The mock logger instance.

    Yields:
        ShellScriptBuilder: A ShellScriptBuilder instance for testing.
    """
    with patch("builtins.open", new_callable=MagicMock), \
         patch("os.makedirs", new_callable=MagicMock), \
         patch("os.chmod", new_callable=MagicMock):
        builder = ShellScriptBuilder(_configuration=mock_configuration, _logger=mock_logger)
        yield builder

def test_build_scripts(shell_script_builder: ShellScriptBuilder) -> None:
    """
    Test the `build_scripts` method to ensure scripts are generated correctly.

    Args:
        shell_script_builder (ShellScriptBuilder): The ShellScriptBuilder instance.
    """
    shell_script_builder.build_scripts()

    test_case_name = list(shell_script_builder.scripts.keys())[0]
    assert test_case_name in shell_script_builder.scripts

    if shell_script_builder.configuration["test_cases"][test_case_name].get("args"):
        num_scripts = (len(shell_script_builder.configuration["test_cases"][test_case_name]["args"]) *
                       len(shell_script_builder.configuration["test_cases"][test_case_name]["datasources"]))
    else:
        num_scripts = len(shell_script_builder.configuration["test_cases"][test_case_name]["datasources"])

    assert len(shell_script_builder.scripts[test_case_name]) == num_scripts

def test_get_datasource_path(shell_script_builder: ShellScriptBuilder) -> None:
    """
    Test the `get_datasource_path` method to ensure it returns the correct path.

    Args:
        shell_script_builder (ShellScriptBuilder): The ShellScriptBuilder instance.
    """
    path = shell_script_builder.get_datasource_path("datasource1")
    expected_path = "/mock/base/path/datasource1/datafile1.txt"
    assert path == expected_path

def test_get_scripts_folder(mock_configuration: dict, mock_logger: MagicMock) -> None:
    """
    Test the `get_scripts_folder` method to ensure it returns the correct folder path.

    Args:
        mock_configuration (dict): The mock configuration dictionary.
        mock_logger (MagicMock): The mock logger instance.
    """
    try:
        builder = ShellScriptBuilder(mock_configuration, mock_logger)
        folder = builder.get_scripts_folder()
        assert folder == os.path.join(os.path.abspath(os.getcwd()), "mock/scripts")
        assert os.path.exists(folder)
    finally:
        shutil.rmtree(folder)

def test_render_command(shell_script_builder: ShellScriptBuilder) -> None:
    """
    Test the `render_command` method to ensure placeholders in commands are rendered correctly.

    Args:
        shell_script_builder (ShellScriptBuilder): The ShellScriptBuilder instance.
    """
    command = "process DATASOURCE with ARG"
    datasource_path = "/mock/path/datafile1.txt"
    rendered_command = shell_script_builder.render_command(
        command, datasource_path, {"pre_commands": ["echo 'Setup'"]}, arg="sample_arg"
    )
    expected_command = "echo 'Setup'\nprocess /mock/path/datafile1.txt with sample_arg"
    assert rendered_command == expected_command

def test_render_test_attributes(shell_script_builder: ShellScriptBuilder) -> None:
    """
    Test the `render_test_attributes` method to ensure known test attributes are replaced correctly.

    Args:
        shell_script_builder (ShellScriptBuilder): The ShellScriptBuilder instance.
    """
    shell_script_builder.configuration["datasources"]["datasource1"]["test_attributes"] = {
        "tnchan": 32, "tdm": "test_tdm", "ttsamp": 20
    }
    command = "run with TNCHAN, TDM, TTSAMP"
    rendered_command = shell_script_builder.render_test_attributes("datasource1", command)
    expected_command = "run with 32, test_tdm, 20"
    assert rendered_command == expected_command

def test_render_tsamp_attribute(shell_script_builder: ShellScriptBuilder) -> None:
    """
    Test the `render_tsamp_attribute` method to ensure 'tsamp' attribute is replaced correctly.

    Args:
        shell_script_builder (ShellScriptBuilder): The ShellScriptBuilder instance.
    """
    shell_script_builder.configuration["datasources"]["datasource1"]["tsamp"] = 6.4e-05
    command = "analyze TSAMP data"
    rendered_command = shell_script_builder.render_tsamp_attribute("datasource1", command)
    expected_command = "analyze 6.4e-05 data"
    assert rendered_command == expected_command

@pytest.mark.parametrize("test_cases_config,expected_script_count", [
    ({"test_cases": {"test1": {"datasources": ["datasource1"], "commands": ["cmd1"]}}}, 1),
    ({"test_cases": {"test1": {"datasources": ["datasource1"], "commands": ["cmd1", "cmd2"]}}}, 2),
    ({"test_cases": {"test1": {"datasources": ["datasource1", "datasource2"], "commands": ["cmd1"]}}}, 2),
    ({"test_cases": {"test1": {"datasources": ["datasource1"], "commands": ["cmd1"], "args": ["arg1", "arg2"]}}}, 2),
])
def test_build_scripts_varied_configs(mock_logger: MagicMock, test_cases_config: dict, expected_script_count: int) -> None:
    """
    Test the `build_scripts` method with varied configurations to ensure correct script generation.

    Args:
        mock_logger (MagicMock): The mock logger instance.
        test_cases_config (dict): The configuration for test cases.
        expected_script_count (int): The expected number of generated scripts.
    """
    config = {
        "base_path": "/mock/base/path",
        "scripts_path": "mock/scripts",
        "datasources": {
            "datasource1": {"file_name": "datafile1.txt"},
            "datasource2": {"file_name": "datafile2.txt"}
        }
    }

    try:
        config.update(test_cases_config)
        shell_script_builder = ShellScriptBuilder(config, mock_logger)
        shell_script_builder.build_scripts()

        test_case_name = list(shell_script_builder.scripts.keys())[0]
        assert len(shell_script_builder.scripts[test_case_name]) == expected_script_count

    finally:
        folder = os.path.join(os.path.abspath(os.getcwd()), "mock/scripts")
        os.path.exists(folder)
        shutil.rmtree(folder)
