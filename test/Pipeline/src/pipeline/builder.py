"""
Python module for building DSPSR functional pipeline test shell scripts

Author: Jesmigel A. Cantos
"""

from __future__ import annotations

import logging
import os

class ShellScriptBuilder:
    """A builder for creating shell scripts used in the DSPSR functional pipeline tests.

    This class generates executable shell scripts for each test case and its associated datasources
    based on a provided configuration. The scripts are written to a specified folder and are designed
    to run the commands defined in the test case configuration.
    """

    def __init__(self: ShellScriptBuilder, _configuration: dict, _logger: logging.Logger | None = None) -> None:
        """
        Initializes the ShellScriptBuilder with a given configuration and logger.

        Args:
            _configuration (dict): The configuration that defines test cases, commands, datasources, and script paths.
            _logger (logging.Logger, optional): A logger instance for logging activities, or None to use the default logger.
        """
        self.logger = _logger or logging.getLogger(__name__)
        self.configuration = _configuration
        self.scripts = {}

    def build_scripts(self: ShellScriptBuilder) -> None:
        """
        Builds shell scripts for all test cases and datasources defined in the configuration.
        """

        for test_case, test_case_config in self.configuration["test_cases"].items():
            self.scripts[test_case] = {}
            for datasource_id in test_case_config["datasources"]:
                datasource_path = self.get_datasource_path(datasource_id)
                self.build_test_case_scripts(test_case, test_case_config, datasource_id, datasource_path)

    def build_test_case_scripts(
        self: ShellScriptBuilder,
        test_case: str,
        test_case_config: dict,
        datasource_id: str,
        datasource_path: str
    ) -> None:
        """
        Builds shell scripts for a specific test case and its associated datasource.

        Args:
            test_case (str): The name of the test case.
            test_case_config (dict): Configuration for the test case, including commands and optional arguments.
            datasource_id (str): The identifier of the datasource.
            datasource_path (str): The path to the datasource file.
        """
        command_number = 0

        for command in test_case_config["commands"]:
            if "args" in test_case_config:
                for arg_number, arg in enumerate(test_case_config["args"]):
                    self.build_script(
                        test_case,
                        test_case_config,
                        datasource_id,
                        datasource_path,
                        command,
                        command_number,
                        arg,
                        arg_number
                    )
            else:
                self.build_script(
                    test_case,
                    test_case_config,
                    datasource_id,
                    datasource_path,
                    command,
                    command_number
                )
            command_number += 1

    def build_script(
        self: ShellScriptBuilder,
        test_case: str,
        test_case_config: dict,
        datasource_id: str,
        datasource_path: str,
        command: str,
        command_number: int,
        arg: str = None,
        arg_number: int = None
    ) -> None:
        """
        Builds a script for a test case command, optionally handling arguments.

        Args:
            test_case (str): The name of the test case.
            test_case_config (dict): Configuration for the test case.
            datasource_id (str): The identifier of the datasource.
            datasource_path (str): The path to the datasource file.
            command (str): The command to be executed.
            command_number (int): The index of the command in the test case.
            arg (str, optional): The argument to be passed to the command (for commands with arguments).
            arg_number (int, optional): The index of the argument (for commands with arguments).
        """
        scripts_folder = self.get_scripts_folder()

        if arg is not None and arg_number is not None:
            script_name = f"{test_case}.{datasource_id}.{command_number}.{arg_number}.sh"
            rendered_command = self.render_command(command, datasource_path, test_case_config, arg)
        else:
            script_name = f"{test_case}.{datasource_id}.{command_number}.sh"
            rendered_command = self.render_command(command, datasource_path, test_case_config)

        self.scripts[test_case][script_name] = {}
        file_path = os.path.join(scripts_folder, script_name)
        self.scripts[test_case][script_name]["file_path"] = file_path

        rendered_command = self.render_test_attributes(datasource_id=datasource_id, command=rendered_command)
        rendered_command = self.render_tsamp_attribute(datasource_id, rendered_command)

        self.create_script_file(file_path, test_case, rendered_command)

        self.scripts[test_case][script_name]["rendered_command"] = rendered_command

    def render_command(self: ShellScriptBuilder, command: str, datasource_path: str, test_case_config: dict, arg: str | None=None) -> str:
        """
        Renders the shell command by replacing placeholders in the command template.

        Args:
            command (str): The command template to be executed.
            datasource_path (str): The path to the datasource file.
            test_case_config (dict): The test case configuration, which may include pre-commands.
            arg (str or None): An optional argument to replace placeholders in the command.

        Returns:
            str: The rendered command with placeholders (e.g., ARG, DATASOURCE) replaced by actual values.
        """

        tokens = command.split()

        if tokens[0] == "dspsr":
            dirname = os.path.dirname(datasource_path)

            parfile = dirname + "/pulsar.par"
            if os.path.exists(parfile):
                command += " -E " + parfile

            predfile = dirname + "/predict.dat"
            if os.path.exists(predfile):
                command += " -P " + predfile

        rendered_command = command

        if "pre_commands" in test_case_config:
            pre_commands = ';\n'.join(test_case_config["pre_commands"])
            rendered_command = f"{pre_commands}\n{rendered_command}"

        if "post_commands" in test_case_config:
            post_commands = ';\n'.join(test_case_config["post_commands"])
            rendered_command = f"{rendered_command}\n{post_commands}"

        if arg:
            rendered_command = rendered_command.replace("ARG", str(arg))

        rendered_command = rendered_command.replace("DATASOURCE", datasource_path)
        return rendered_command

    def create_script_file(self: ShellScriptBuilder, file_path: str, test_case: str, command_content: str) -> None:
        """
        Creates a shell script file with the specified content and sets it as executable.

        Args:
            file_path (str): The full path to the script file to be created.
            test_case (str): The test case ID
            command_content (str): The content (commands) to be written to the script file.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        comment = f"# DO NOT EDIT! This file is automatically generated for {test_case}"
        with open(file_path, 'w') as script_file:
            script_file.write(f"#!/bin/bash\n{comment}\n{command_content}\n")

        os.chmod(file_path, 0o755)

    def get_datasource_path(self: ShellScriptBuilder, datasource_id: str) -> str:
        """
        Retrieves the file path for the given datasource.

        Args:
            datasource_id (str): The identifier of the datasource.

        Returns:
            str: The full path to the datasource file, built from the base path and datasource ID.
        """
        base_path = self.configuration["base_path"]
        file_name = self.configuration["datasources"][datasource_id]["file_name"]
        return f"{base_path}/{datasource_id}/{file_name}"

    def get_scripts_folder(self: ShellScriptBuilder) -> str:
        """
        Retrieves the folder where the shell scripts will be saved.
        The folder is created if it does not exist.

        Returns:
            str: The full path to the scripts folder.
        """
        scripts_path = self.configuration["scripts_path"]
        current_directory = os.path.abspath(os.getcwd())
        scripts_folder = os.path.join(current_directory, scripts_path)

        # Create the folder only if it doesn't already exist
        if not os.path.exists(scripts_folder):
            os.makedirs(scripts_folder)

        return scripts_folder

    def render_test_attributes(self: ShellScriptBuilder, datasource_id: str, command: str) -> str:
        """
        Replaces known test attribute placeholders in the command with their actual values.

        Args:
            datasource_id (str): The identifier of the datasource.
            command (str): The command in which to replace attribute placeholders.

        Returns:
            str: The command with replaced test attributes.
        """
        known_test_attributes = ["TNCHAN", "TDM", "TTSAMP"]
        for attribute in known_test_attributes:
            if attribute in command:
                value = self.configuration["datasources"][datasource_id]["test_attributes"][attribute.lower()]
                command = command.replace(attribute, str(value))

        return command

    def render_tsamp_attribute(self: ShellScriptBuilder, datasource_id: str, command: str) -> str:
        """
        Detects and replaces the 'tsamp' attribute in the command if found as a whole word.

        Args:
            datasource_id (str): The identifier of the datasource.
            command (str): The command to check for 'tsamp' keyword.

        Returns:
            str: The command with 'tsamp' replaced by its corresponding value.
        """
        # Split the command by spaces to detect 'tsamp' as a whole word and not as part of 'ttsamp'
        words = command.split(" ")
        for i, word in enumerate(words):
            if word == "TSAMP":
                tsamp_value = self.configuration["datasources"][datasource_id].get("tsamp")
                if tsamp_value:
                    words[i] = str(tsamp_value)

        # Join the words back into a single string
        return " ".join(words)
