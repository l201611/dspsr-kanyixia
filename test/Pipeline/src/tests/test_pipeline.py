"""
TODO:
- [x] Develop function that converts config.yaml into a list of tests
- [x] Develop fixture that goes through all of test executions
- [] Develop test code that goes through functional pipeline tests
- [] Develop test code that goes through failed functional pipeline tests with suggested memory configurations
- [] Develop test code that goes through functional pipeline tests results and perform diff against baseline results
"""

from unittest.mock import MagicMock, patch, mock_open, ANY
from pipeline import Orchestrator
import pytest
import shutil
import yaml

def helper_orchestrator()->Orchestrator:
    """
    Helper function to create and set up an instance of the Orchestrator class.

    This function initializes the Orchestrator with a dry run mode and a specific
    payload YAML path, then calls its `build_scripts` method to prepare the scripts.

    Returns:
        Orchestrator: A configured instance of the Orchestrator class.
    """
    _orchestrator = Orchestrator(_dry_run=True, _payload_yaml_path="config.yaml")
    _orchestrator.build_scripts()
    return _orchestrator

def helper_test_case_list() -> list:
    """
    Helper function to retrieve a list of test cases from the Orchestrator's builder scripts.

    This function uses the helper_orchestrator() to get the orchestrator instance,
    accesses its builder's `scripts` dictionary, and extracts the test case names.

    Returns:
        list: A list of test case names available in the builder's scripts.
    """
    result = [test_case for test_case, files in helper_orchestrator().builder.scripts.items()]
    return result

@pytest.fixture(params=helper_test_case_list())
def payload_test_case(request: pytest.FixtureRequest)->str:
    """
    Pytest fixture to provide individual test case names as parameters to tests.

    This fixture generates test cases using the `helper_test_case_list` function
    and makes them available for parameterized testing.

    Args:
        request (pytest.FixtureRequest): The pytest fixture request object.

    Returns:
        str: A single test case name from the list of available test cases.
    """
    return request.param

@pytest.fixture
def orchestrator()->Orchestrator:
    """
    Pytest fixture to provide a pre-configured instance of the Orchestrator class.

    This fixture calls the `helper_orchestrator` function to create and set up
    the Orchestrator instance for use in tests.

    Returns:
        Orchestrator: A configured instance of the Orchestrator class.
    """
    return helper_orchestrator()

def test_testcases_loaded_to_builder(orchestrator, payload_test_case)->None:
    """
    Test to verify that all test cases are correctly loaded into the Orchestrator's builder.

    This test checks if the given payload test case is present in the builder's scripts,
    and cleans up the associated scripts folder after the test.

    Args:
        orchestrator (Orchestrator): A pytest fixture providing the Orchestrator instance.
        payload_test_case (str): A pytest fixture providing the name of a test case.
    """
    assert payload_test_case in orchestrator.builder.scripts.keys()
    folder = orchestrator.builder.get_scripts_folder()
    shutil.rmtree(folder)