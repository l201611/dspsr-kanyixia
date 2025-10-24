"""
Entrypoint code of DSPSR functional pipeline test framework
"""

import argparse
import logging
import re
import signal
import sys

from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse the command line arguments for the DSPSR pipeline test controller.

    Arguments:
        --config: Path to the config (YAML) file containing DSPSR pipeline test configuration (required).
        -v, --verbose: Increase verbosity level (-v, -vv, -vvv) (optional).
        --test_case_id: The specific test case ID to be executed (optional).

    Returns:
        A tuple of parsed arguments and any additional unknown arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="DSPSR pipeline test controller",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config (YAML) file containing DSPSR pipeline test configuration.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (-v).",
    )
    parser.add_argument(
        "--test_case_id",
        type=str,
        help="The specific test case ID to be executed. If not provided, all test cases are executed.",
    )
    parser.add_argument(
        "--marker",
        type=str,
        help="Marker filename  created upon output folder creation.",
    )
    parser.add_argument(
        "--dry-run",
        type=bool,
        default=False,
        help="Print rendered commands used by the test cases. Does not execute the commands.",
    )
    parser.add_argument(
        "--nthread",
        type=int,
        default=1,
        help="Number of threads with which to execute commands.",
    )

    return parser.parse_known_args()


def main():
    """DSPSR pipeline main function."""
    if __name__ == "__main__":
        args, _ = parse_args()

        # Setting verbosity levels
        # 10: DEBUG
        # 20: INFO
        # 40: ERROR
        # 50: CRITICAL
        logging_level = max(10, logging.INFO - 10 * args.verbose)

        log_format = "%(asctime)s : %(levelname)5s : %(filename)s:%(lineno)s %(funcName)s() : %(msg)s"
        logging.basicConfig(format=log_format, level=logging_level)

        payload_yaml_path = args.config
        marker_filename = args.marker
        dry_run = args.dry_run
        nthread = args.nthread
        logger = logging.getLogger(__name__)

        orchestrator = Orchestrator(_payload_yaml_path=payload_yaml_path, _dry_run=dry_run, _marker_filename=marker_filename, _nthread=nthread, _logger=logger)

        # Handle signals
        signal.signal(signal.SIGTERM, orchestrator.stop)
        signal.signal(signal.SIGINT, orchestrator.stop)
        signal.signal(signal.SIGHUP, orchestrator.stop)

        # Start the orchestrator with optional test case ID
        if args.test_case_id:
            logger.debug(f"Starting orchestrator for test case ID: {args.test_case_id}")
            orchestrator.start(test_case_id=args.test_case_id)
        else:
            logger.debug("Starting orchestrator for all test cases.")
            orchestrator.start()

        sys.exit()

main()
