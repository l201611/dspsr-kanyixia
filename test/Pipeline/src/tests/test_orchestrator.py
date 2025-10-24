"""
Orchestrator test suite
"""

import datetime
import logging
import os
import pytest
import random
import shutil
import signal
import subprocess
import yaml

from typing import Any, Generator, Optional, Dict, Union
from unittest.mock import MagicMock, patch, mock_open, ANY

from pipeline import Orchestrator

def clean_environment(orchestrator: Orchestrator)->None:
    folder = orchestrator.builder.get_scripts_folder()
    shutil.rmtree(folder)

@pytest.fixture
def configuration_payload()->str:
    return "config.yaml"

@pytest.fixture
def configuration_contents(configuration_payload: str)->yaml.YAMLObject:
    with open(configuration_payload, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)

    return yaml_data

@pytest.fixture(params=[False, True])
def _dry_run(request: pytest.FixtureRequest)->bool:
    """Fixture to set dry run mode on or off."""
    return request.param

@pytest.fixture(params=[None, "test"])
def _marker_filename(request: pytest.FixtureRequest)->Any:
    """Fixture to provide different marker filenames."""
    return request.param

@pytest.fixture(params=[None, "TC01", "TC03"])
def _test_case_id(request: pytest.FixtureRequest)->Any:
    """Fixture to provide different test case id's."""
    return request.param

@pytest.fixture
def mock_logger()->MagicMock:
    return MagicMock()

@pytest.fixture
def orchestrator(
        mock_logger: MagicMock,
        configuration_payload: str,
        _dry_run: bool,
        _marker_filename: Optional[str],
        _test_case_id: Optional[str],
    ) -> Orchestrator:
    """Fixture to create an Orchestrator instance with various configurations."""
    return Orchestrator(
        _payload_yaml_path=configuration_payload,
        _dry_run=_dry_run,
        _logger=mock_logger,
        _marker_filename=_marker_filename
    )

def test_configuration_loaded(orchestrator: Orchestrator, configuration_contents: yaml.YAMLObject) -> None:
    """Test the configuration loaded to the script builder through the orchestrator."""
    try:
        orchestrator.build_scripts()
        assert configuration_contents["test_cases"].keys() == orchestrator.builder.scripts.keys()
    finally:
        clean_environment(orchestrator)

def test_orchestrator_start(orchestrator: Orchestrator, _dry_run: bool, _test_case_id: str)->None:
    """Test the orchestrator's start method in dry-run, regular modes paired with valid testcase id's."""
    try:
        with patch("pipeline.Orchestrator.prepare_output", new_callable=MagicMock) as mock_prepare_output, \
            patch("pipeline.Orchestrator.execute_script", new_callable=MagicMock) as mock_execute_script, \
            patch("pipeline.Orchestrator.append_to_log", new_callable=MagicMock) as mock_append_to_log:
            
            orchestrator.start(test_case_id=_test_case_id)

            if _dry_run:
                # In dry-run mode, ensure neither prepare_output nor execute_script are called
                mock_prepare_output.assert_not_called()
                mock_execute_script.assert_not_called()
            else:
                # In regular mode, verify calls and print details
                # Print each call's details for `prepare_output` to stdout
                for idx, call in enumerate(mock_prepare_output.call_args_list, start=1):
                    print(f"Call {idx}: {call}")
                
                # Print each call's details for `execute_script` to stdout
                for idx, call in enumerate(mock_execute_script.call_args_list, start=1):
                    print(f"Call {idx}: {call}")

                expected_call_count = 0
                if _test_case_id:
                    expected_call_count = len(orchestrator.builder.scripts[_test_case_id].items())
                else:
                    for test_case, test_case_configurations in orchestrator.builder.scripts.items():
                        expected_call_count += len(test_case_configurations.keys())

                print(f"Total test executions: {expected_call_count}")
                
                # Assert the expected number of calls
                assert mock_prepare_output.call_count == expected_call_count
                assert mock_execute_script.call_count == expected_call_count
    finally:
        clean_environment(orchestrator)

def test_orchestrator_start_invalid_test_case_id(mock_logger: MagicMock, orchestrator: Orchestrator, _dry_run: bool)->None:
    try:
        orchestrator._dry_run = _dry_run
        with patch("pipeline.Orchestrator.prepare_output", new_callable=MagicMock) as mock_prepare_output, \
            patch("pipeline.Orchestrator.execute_script", new_callable=MagicMock) as mock_execute_script:
            invalid_id = "TCINVALID"
            orchestrator.start(test_case_id=invalid_id)
        mock_logger.error.assert_any_call(f"Test case ID '{invalid_id}' does not exist in the configuration.")
    finally:
        clean_environment(orchestrator)

def test_append_failed_test_case(mock_logger: MagicMock, configuration_payload: str, _test_case_id: str)->None:
    """Test execute_script returns False on a subprocess error, indicating a failed test case execution."""

    try:
        orchestrator = Orchestrator(_logger=mock_logger, _payload_yaml_path=configuration_payload)

        # Setup mocks for prepare_output, subprocess.run, os.makedirs, os.chmod, and open
        with patch.object(orchestrator, "prepare_output", new_callable=MagicMock) as mock_prepare_output, \
            patch("subprocess.run") as mock_subprocess_run, \
            patch("os.makedirs") as mock_makedirs, \
            patch("os.chmod") as mock_chmod, \
            patch("builtins.open", mock_open()) as mock_file:

            # Mock subprocess.run to simulate a runtime error with exit code -1
            mock_subprocess_run.side_effect = subprocess.CalledProcessError(returncode=-1, cmd="mock command")

            # Mock os.makedirs and os.chmod to avoid filesystem changes
            mock_makedirs.return_value = None
            mock_chmod.return_value = None

            if _test_case_id is not None:
                orchestrator.build_scripts()
                testcase_config = orchestrator.builder.scripts[_test_case_id]
                results_path = "dummy/results_path"

                # Call execute_scripts and assert that it returns False due to the mocked error
                failed_cases = orchestrator._execute_test_case(
                    test_case_id=_test_case_id,
                    testcase_config=testcase_config
                )

                # Check if any of the failed cases has the 'testcase' value as _test_case_id
                assert any(failure["testcase"] == _test_case_id for failure in failed_cases), \
                    f"Test case '{_test_case_id}' should have been appended to the failed cases list due to execution error"
    finally:
        # Clean up resources after the test
        clean_environment(orchestrator)

def test_prepare_output(orchestrator: Orchestrator, mock_logger: MagicMock)->None:
    """Test prepare_output creates the necessary directory and marker file with correct permissions."""

    try:
        # Initialize Orchestrator instance with necessary attributes
        orchestrator.marker_filename = "marker.txt"
        
        test_destination = "dummy/test_destination"
        test_results_base = orchestrator.test_results_base

        # Patch os.makedirs and open to avoid file system changes
        with patch("os.makedirs") as mock_makedirs, \
            patch("builtins.open", mock_open()) as mock_file:

            # Call the prepare_output method
            orchestrator.prepare_output(test_destination)

            # Assert that os.makedirs was called with the correct path and mode
            mock_makedirs.assert_called_once_with(test_destination, mode=0o764, exist_ok=True)

            # Check if the marker file was created at the correct location
            expected_marker_file_path = f"{test_results_base}/{orchestrator.marker_filename}"
            mock_file.assert_called_once_with(expected_marker_file_path, 'w')

    finally:
        # Clean up resources after the test
        clean_environment(orchestrator)

def test_results_summary_with_failures(mock_logger: MagicMock, orchestrator: Orchestrator)->None:
    """Test that results_summary logs the correct information for failed test cases."""
    
    # Sample failed cases to simulate execution errors
    failed_cases = [
        {
            "testcase": "TC01",
            "command": "mock_command_1",
            "log_file": "mock_path_1/execution_log.txt"
        },
        {
            "testcase": "TC02",
            "command": "mock_command_2",
            "log_file": "mock_path_2/execution_log.txt"
        }
    ]
    
    # Call results_summary with failed cases
    orchestrator.failed_cases = failed_cases
    orchestrator.results_summary()
    
    # Check that the logger was called with a formatted summary table
    assert mock_logger.info.call_count > 0
    log_output = "\n".join(call.args[0] for call in mock_logger.info.call_args_list)
    assert "Failed Test Cases Summary:" in log_output
    assert "TC01" in log_output
    assert "mock_command_1" in log_output
    assert "mock_path_1/execution_log.txt" in log_output
    assert "TC02" in log_output
    assert "mock_command_2" in log_output
    assert "mock_path_2/execution_log.txt" in log_output

def test_results_summary_no_failures(mock_logger: MagicMock, orchestrator: Orchestrator)->None:
    """Test that results_summary logs a success message when there are no failures and dry_run is False."""

    orchestrator.dry_run = False
    
    # Call results_summary with an empty failed_cases list
    orchestrator.results_summary()
    
    # Assert the logger was called with a success message
    mock_logger.info.assert_called_once_with("All test cases passed successfully.")

def test_stop_method(mock_logger: MagicMock, orchestrator: Orchestrator)->None:
    """Test the stop method in the Orchestrator class."""
    # Patch os.makedirs and open to avoid file system changes
    with patch("os.makedirs") as mock_makedirs, \
         patch("builtins.open", mock_open()) as mock_file:

        # Initialize the Orchestrator with the mock logger
        mock_payload_yaml_path = "dummy/path/to/config.yaml"
        
        # Simulate signal and frame
        test_signum = signal.SIGINT  # or another signal, e.g., SIGTERM
        test_frame = None  # You can also mock a frame if necessary

        # Call the stop method
        orchestrator.stop(signum=test_signum, frame=test_frame)
        
        # Check if logger.info was called with the correct message
        mock_logger.info.assert_called_once_with(f"Stopping the Orchestrator due to signal {test_signum} and frame {test_frame}")

def test_execute_script_success(orchestrator: Orchestrator)->None:
    # Mock subprocess.run to simulate successful script execution
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = None  # Simulate successful run (no exceptions)
        
        # Mock the creation of the log file and directory
        with patch("builtins.open", mock_open()) as mock_file, \
             patch("os.path.join", return_value='/path/to/results/execution_log.txt'), \
             patch("os.path.isdir", return_value=True), \
             patch("pipeline.Orchestrator.append_to_log", new_callable=MagicMock) as mock_append_to_log:  # Mock that the directory exists
            
            test_script = "/path/to/test_script.sh"
            results_path = "/path/to/results"
            
            # Call the method
            result = orchestrator.execute_script(test_script, results_path)

            # Assert subprocess.run was called with correct arguments
            mock_run.assert_called_once_with(
                [test_script],
                check=True,
                cwd=results_path,
                stdout=ANY,
                stderr=ANY,
                timeout=600
            )
            
            # Assert the method returns True for success
            assert result is True

            # Ensure the log file was opened
            mock_file.assert_any_call('/path/to/results/execution_log.txt', 'a')

def test_execute_script_failure(orchestrator: Orchestrator)->None:
    # Mock subprocess.run to raise a CalledProcessError
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, '/path/to/test_script.sh')

        # Mock the creation of the log file and directory
        with patch("builtins.open", mock_open()) as mock_file, \
             patch("os.path.join", return_value='/path/to/results/execution_log.txt'), \
             patch("os.path.isdir", return_value=True):  # Mock that the directory exists
            
            test_script = "/path/to/test_script.sh"
            results_path = "/path/to/results"
            
            # Call the method
            result = orchestrator.execute_script(test_script, results_path)
            
            # Assert subprocess.run was called with correct arguments
            mock_run.assert_called_once_with(
                [test_script],
                check=True,
                cwd=results_path,
                stdout=ANY,
                stderr=ANY,
                timeout=600
            )
            
            # Assert the method returns False when an error occurs
            assert result is False

            # Ensure the log file was opened
            mock_file.assert_any_call('/path/to/results/execution_log.txt', 'a')