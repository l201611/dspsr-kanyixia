"""
Python module for DSPSR functional pipeline test orchestration

Author: Jesmigel A. Cantos
"""
from __future__ import annotations
from typing import Optional
from tabulate import tabulate
import datetime
import logging
import os
import glob
import re
import shlex
import subprocess
import yaml

from .builder import ShellScriptBuilder

class Orchestrator:
    """
    Orchestrator class for managing the DSPSR functional pipeline tests.

    This class builds and executes shell scripts for DSPSR tests,
    capturing any failed test cases and providing a summary.
    """

    def __init__(
        self: Orchestrator,
        _payload_yaml_path: str,
        _dry_run: bool = False,
        _marker_filename: Optional[str] = None,
        _nthread: int = 1,
        _logger: logging.Logger | None = None
    ) -> None:
        """
        Initialize the Orchestrator instance.

        Args:
            _payload_yaml_path (str): Path to the YAML configuration file.
            _dry_run (bool, optional): If True, perform a dry run without actual execution. Defaults to False.
            _marker_filename (Optional[str], optional): Name of the marker file to create in the output directory.
            _logger (logging.Logger | None, optional): Logger instance to use. Defaults to None.
        """
        self.abort = False
        self.nthread = _nthread
        self.logger = _logger or logging.getLogger(__name__)
        self.configuration = self.load_yaml(payload_yaml_path=_payload_yaml_path)
        self.builder = ShellScriptBuilder(_configuration=self.configuration)
        self.dry_run = _dry_run
        self.marker_filename = _marker_filename
        self.test_results_base = os.path.join(
            self.configuration["base_path"],
            f"tests/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        )
        self.failed_cases = []  # Store failed test case details across executions

    def start(self: Orchestrator, test_case_id: Optional[str] = None) -> None:
        """
        Start the orchestration process by executing DSPSR test cases.

        Args:
            test_case_id (Optional[str]): Specific test case ID to execute. If not provided, all test cases are executed.
        """
        self.logger.info(f"Starting DSPSR test orchestration for {'all test cases' if test_case_id is None else f'test case ID: {test_case_id}'}")
        self.build_scripts()
        self.execute_scripts(test_case_id=test_case_id)

    def stop(self: Orchestrator, signum=None, frame=None) -> None:
        """
        Stop the orchestrator and log the stop event with signal information.

        Args:
            signum (int, optional): The signal number that triggered the stop.
            frame (signal frame, optional): The current stack frame when the signal was received.
        """
        self.logger.info(f"Stopping the Orchestrator due to signal {signum} and frame {frame}")
        self.abort = True

    def build_scripts(self: Orchestrator) -> None:
        """
        Build the DSPSR test scripts using the ShellScriptBuilder instance.
        """
        self.logger.info("Starting to build DSPSR test scripts")
        self.builder.build_scripts()
        self.logger.debug("Completed building DSPSR test scripts")

    def execute_scripts(self: Orchestrator, test_case_id: Optional[str] = None) -> None:
        """
        Execute the DSPSR test scripts and log failed cases.

        Args:
            test_case_id (Optional[str]): Specific test case ID to execute. If not provided, all test cases are executed.
        """
        self.logger.info("Executing DSPSR test scripts")
        self.failed_cases.clear()

        if test_case_id:
            # Execute a single test case
            if test_case_id not in self.builder.scripts:
                self.logger.error(f"Test case ID '{test_case_id}' does not exist in the configuration.")
                return

            testcase_config = self.builder.scripts[test_case_id]
            self.failed_cases.extend(self._execute_test_case(test_case_id, testcase_config))
        else:
            # Execute all test cases
            for test_case_id, testcase_config in self.builder.scripts.items():
                self.failed_cases.extend(self._execute_test_case(test_case_id, testcase_config))
                if self.abort:
                    break

        # Print a summary of test results
        self.results_summary()

    def _execute_test_case(self: Orchestrator, test_case_id: str, testcase_config: dict) -> list:
        """
        Execute a specific test case and capture any failures.

        Args:
            test_case_id (str): The ID of the test case.
            testcase_config (dict): The configuration dictionary for the test case.

        Returns:
            list: A list of failed test cases, with details for each failure.
        """
        header = f"\nTest case: {test_case_id}\nDescription: {self.configuration['test_cases'][test_case_id]['description']}"
        self.logger.info(header)
        failed_cases = []

        for script_name, script_configuration in testcase_config.items():
            test_destination = os.path.join(self.test_results_base, script_name)
            log_file_path = os.path.join(test_destination, 'execution_log.txt')
            rendered_command = script_configuration["rendered_command"]

            # save command without -threads for use on second try, if needed
            backup_command = rendered_command

            if self.nthread > 1:
                rendered_command = f"{rendered_command} -threads {self.nthread}"

            message = f"Executing script {script_name} for test case {test_case_id}"
            if self.dry_run:
                self.logger.info(f"dry_run set to {self.dry_run}: * Not * {message}")
                continue

            self.logger.info(message)
            message = f"Rendered command:\n{rendered_command}\n"
            self.logger.info(message)
            self.prepare_output(test_destination)
            self.append_to_log(test_destination, f"# {message}\n")

            if self.execute_script(rendered_command, test_destination):
                if self.abort:
                    break
                self.compare_results(test_case_id=test_case_id, results_path=test_destination)
                continue

            if self.abort:
                break

            # Attempt re-execution if a suggested argument is found
            suggested_cli_arg = self.check_log_for_suggested_argument(test_destination)
            if suggested_cli_arg:
                # To avoid having multiple threads using an arbitrarily large amount of memory,
                # use command that was backed up before any "-threads" argument may have been added.
                rendered_command = f"{backup_command} {suggested_cli_arg}"
                message = "Re-executing with suggested argument"
                if self.nthread > 1:
                    message += " and without multi-threading"
                message += f":\n{rendered_command}\n"
                self.logger.info(message)
                self.append_to_log(test_destination, f"\n# {message}\n")
                if self.execute_script(rendered_command, test_destination):
                    if self.abort:
                        break
                    self.compare_results(test_case_id=test_case_id, results_path=test_destination)
                    continue

            if self.abort:
                break

            # Record the failure
            failed_cases.append({
                'testcase': test_case_id,
                'reason': "execution fail",
                'script': script_name,
                'command': rendered_command,
                'log_file': log_file_path
            })

        return failed_cases

    def append_to_log(self: Orchestrator, test_destination: str, message: str) -> None:
        """
        Append a message to the log file at the specified test destination.

        Args:
            test_destination (str): Directory for the log file.
            message (str): Message to append to the log.
        """
        log_file_path = os.path.join(test_destination, 'execution_log.txt')
        with open(log_file_path, 'a') as log_file:
            log_file.write(message)

    def check_log_for_suggested_argument(self: Orchestrator, test_destination: str) -> Optional[str]:
        """
        Check the log file for a suggested CLI argument for retrying execution.

        Args:
            test_destination (str): Directory of the log file.

        Returns:
            Optional[str]: Suggested CLI argument, if found; otherwise None.
        """
        log_file_path = os.path.join(test_destination, 'execution_log.txt')
        if not os.path.exists(log_file_path):
            self.logger.error(f"Log file not found at {log_file_path}")
            return None

        with open(log_file_path, 'r', encoding='utf-8') as log_file:
            log_contents = log_file.read()

        match = re.search(r'.*minimum of\s+"([^"]+)"\s+on command line.*', log_contents)
        if match:
            return match.group(1)

        return None

    def prepare_output(self: Orchestrator, test_destination: str) -> None:
        """
        Prepare the output directory for test results and create an optional marker file.

        Args:
            test_destination (str): Path to create output files.
        """
        os.makedirs(test_destination, mode=0o755, exist_ok=True)
        if self.marker_filename:
            marker_file_path = os.path.join(self.test_results_base, self.marker_filename)
            with open(marker_file_path, 'w') as marker_file:
                marker_file.write(f"Marker file created at {datetime.datetime.now()}\n")

    def execute_script(self: Orchestrator, test_command: str, results_path: str) -> bool:
        """
        Execute a shell command and log output to a file.

        Args:
            test_command (str): The command to execute.
            results_path (str): Path where log output will be saved.

        Returns:
            bool: True if the command succeeded; False otherwise.
        """
        log_file = os.path.join(results_path, 'execution_log.txt')
        script_file = os.path.join(results_path, 'test.sh')
        with open(script_file, "w") as file:
            file.write(test_command)
        try:
            with open(log_file, 'a') as f:
                subprocess.run(['sh', './test.sh'], check=True, cwd=results_path, stdout=f, stderr=f, timeout=600)
            return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            self.logger.error(f"Failed to execute command: {test_command}\nLog: {log_file}")
            return False

    def compare_results(self: Orchestrator, test_case_id: str, results_path: str)->None:
        """
        Compare generated files in results_path with their baseline counterparts.

        Args:
            test_case_id (str): The test case ID.
            results_path (str): Directory where the test results are logged.
        """
        self.logger.debug(f"results_path: {results_path}")
        log_file_path = os.path.join(results_path, 'execution_log.txt')
        script_name = results_path.split("/")[-1]
        self.logger.debug(f"script_name: {script_name}")
        baseline_path = os.path.join(
            self.configuration["base_path"],
            f"tests/baseline/{script_name}"
        )
        self.logger.info(f"Baseline Path: {baseline_path}")

        # Find all .ar and .sf files in the results_path
        ar_files = glob.glob(os.path.join(results_path, "*.ar"))
        sf_files = glob.glob(os.path.join(results_path, "*.sf"))
        command = ""

        try:
            # Compare .ar files using psrdiff
            for test_file in ar_files:
                baseline_file = os.path.join(baseline_path, os.path.basename(test_file))
                if os.path.exists(baseline_file):
                    command = f"psrdiff -X {baseline_file} {test_file}"
                    self.logger.info(f"comparison command: {command}")

                    result = subprocess.run(shlex.split(command), check=True, text=True, capture_output=True)
                    if not self.analyse_diff_output(command_name="psrdiff", command_output=result.stdout):
                        self.failed_cases.append({
                            'testcase': test_case_id,
                            'reason': "psrdiff fail",
                            'script': script_name,
                            'command': command,
                            'log_file': log_file_path
                        })
                else:
                    self.logger.warning(f"Baseline file not found for {test_file}")

            # Compare .sf files using diff
            for test_file in sf_files:
                baseline_file = os.path.join(baseline_path, os.path.basename(test_file))
                if os.path.exists(baseline_file):
                    command = f"digidiff -X {baseline_file} {test_file}"
                    self.logger.info(f"comparison command: {command}")

                    result = subprocess.run(shlex.split(command), check=True, text=True, capture_output=True)
                    if not self.analyse_diff_output(command_name="digidiff", command_output=result.stdout):
                        self.failed_cases.append({
                            'testcase': test_case_id,
                            'reason': "digidiff fail",
                            'script': script_name,
                            'command': command,
                            'log_file': log_file_path
                        })
                else:
                    self.logger.warning(f"Baseline file not found for {test_file}")
        except Exception as e:
            # TODO: this is a brute force error handling. Refactor later
            self.logger.error(f"Error during comparison execution: {e}")
            self.failed_cases.append({
                'testcase': test_case_id,
                'reason': "exception caught",
                'script': script_name,
                'command': command,
                'log_file': log_file_path
            })


    def analyse_diff_output(self, command_name: str, command_output: str, threshold: int = 1) -> bool:
        """
        Analyse the output of psrdiff or digidiff to check if 'chisq' value is greater than 0.

        Args:
            command_name (str): The name of the executable used for comparison ('digidiff' or 'psrdiff').
            command_output (str): The output of the command to analyze.
            threshold (int): The threshold used to determine if chisq is acceptable.

        Returns:
            bool: True if 'chisq' is greater than 1, False otherwise.
        """
        if command_name == "digidiff":
            try:
                value = float(command_output.strip())
                self.logger.debug(f"Extracted chisq value from digidiff: {value}")
            except ValueError:
                self.logger.error("Unable to parse chisq from digidiff output as a float.")
                raise ValueError("Invalid output from digidiff.")

        elif command_name == "psrdiff":
            try:
                value = float(command_output.split(" ")[2])
                self.logger.debug(f"Extracted chisq value from psrdiff: {value}")
            except ValueError:
                self.logger.error("Unable to parse chisq from psrdiff output as a float.")
                raise ValueError("Invalid output from psrdiff.")

        else:
            self.logger.error(f"Unknown tool name: {command_name}")
            raise ValueError(f"Unsupported tool: {command_name}")

        if value > threshold:
            self.logger.warn(f"Extracted chisq value: {value} > threshold: {threshold}")
            return False
        else:
            self.logger.info(f"Extracted chisq value: {value} < threshold: {threshold}")
            return True


    def results_summary(self) -> None:
        """
        Print a summary of test results, listing failed test cases if any.
        """

        if self.abort:
            return

        if self.failed_cases:
            self.logger.info("Failed Test Cases Summary:\n" + tabulate(self.failed_cases, headers="keys"))
        else:
            self.logger.info("All test cases passed successfully.")


    def load_yaml(self: Orchestrator, payload_yaml_path: str = "") -> yaml.YAMLObject:
        """
        Load a YAML configuration file.

        Args:
            payload_yaml_path (str): Path to the YAML file.

        Returns:
            yaml.YAMLObject: Parsed YAML configuration object.
        """
        with open(payload_yaml_path, 'r') as file:
            return yaml.safe_load(file)
