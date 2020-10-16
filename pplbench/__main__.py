# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import sys

from .main import main


def console_main() -> None:
    """The entry point for CLI with special handling for BrokenPipError. This function is
    not intended to be called as a function from another program."""
    try:
        main()
        sys.stdout.flush()
        sys.stderr.flush()
    except BrokenPipeError:
        # Python flushes standard streams on exit; redirect remaining output
        # to devnull to avoid another BrokenPipeError at shutdown
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(1)  # Python exits with error code 1 on EPIPE


if __name__ == "__main__":
    console_main()
