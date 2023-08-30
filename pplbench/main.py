# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from types import SimpleNamespace
from typing import List, Optional
from unittest import mock

from jsonargparse import ActionJsonSchema, ArgumentParser, dict_to_namespace

from .lib import model_helper, ppl_helper, reports, utils


# The following schema defines the experiment that PPLBench should run.
SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "object",
            "properties": {
                "class": {"type": "string"},
                "args": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "minimum": 2},
                        "k": {"type": "integer", "minimum": 1},
                    },
                    "required": ["n"],
                    "additionalProperties": True,
                    "propertyNames": {"pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
                },
                "seed": {"type": "integer", "minimum": 1},
                "package": {"type": "string"},  # defaults to pplbench.models
            },
            "required": ["class"],
            "additionalProperties": False,
        },
        "iterations": {"type": "integer", "minimum": 1},
        "num_warmup": {"type": "integer", "minimum": 0},
        "trials": {"type": "integer", "minimum": 2},
        "profile_run": {"type": "boolean"},
        "num_profiled": {"type": "integer"},
        "strip_profiled_names": {"type": "boolean"},
        "profiling_tools_dir": {"type": "string"},
        "profiling_type": {"type": "string", "enum": ["deterministic", "statistical"]},
        "ppls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "inference": {
                        "type": "object",
                        "properties": {
                            "class": {"type": "string"},
                            "num_warmup": {"type": "integer", "minimum": 0},
                            "compile_args": {
                                "type": "object",
                                "propertyNames": {
                                    "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"
                                },
                                "additionalProperties": True,
                            },
                            "infer_args": {
                                "type": "object",
                                "propertyNames": {
                                    "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"
                                },
                                "additionalProperties": True,
                            },
                        },
                        "required": ["class"],
                        "additionalProperties": False,
                    },
                    "legend": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "color": {"type": "string"},
                        },
                    },
                    "seed": {"type": "integer", "minimum": 1},
                    # package defaults to pplbench.ppls.<ppl name>
                    "package": {"type": "string"},
                },
                "required": ["name", "inference"],
                "additionalProperties": False,
            },
        },
        "loglevels": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            },
        },
        "save_samples": {"type": "boolean"},  # default "false"
        "output_root_dir": {"type": "string"},  # defaults to "./outputs"
        "figures": {
            "type": "object",
            "properties": {
                "generate_pll": {"type": "boolean"},  # default "true"
                "suffix": {"type": "string"},  # default "png"
            },
            "additionalProperties": False,
        },
    },
    "required": ["model", "ppls", "iterations", "trials"],
    "additionalProperties": False,
}

LOGGER = logging.getLogger("pplbench")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def main(args: Optional[List[str]] = None) -> None:
    # first load the configuration, start logging and find all needed classes
    config = read_config(args)
    output_dir = utils.create_output_dir(config)
    configure_logging(config, output_dir)
    # only import profiler if needed
    if hasattr(config, "profile_run") and config.profile_run:
        from .lib.ppl_profiler import PPL_Profiler

        ppl_profiler = PPL_Profiler(output_dir, config)
    model_cls = model_helper.find_model_class(config.model)
    all_ppl_details = ppl_helper.find_ppl_details(config)
    # then start the actual benchmarking run
    train_data, test_data = model_helper.simulate_data(config.model, model_cls)
    (
        all_variable_metrics_data,
        all_other_metrics_data,
    ) = ppl_helper.collect_samples_and_stats(
        config, model_cls, all_ppl_details, train_data, test_data, output_dir
    )
    # finally, output charts
    reports.generate_plots(
        output_dir,
        config,
        all_ppl_details,
        all_variable_metrics_data,
        all_other_metrics_data,
    )
    if hasattr(config, "profile_run") and config.profile_run:
        ppl_profiler.finish_profiling()

    from .ppls.beanmachine.graph.inference import bmg_stats

    if bmg_stats is not None:
        stat_filename = output_dir + "/BMG_stats.txt"
        with open(stat_filename, "w") as stat_file:
            stat_file.write(str(bmg_stats))

    # The last output should be the name of the directory
    LOGGER.info(f"Output saved in '{output_dir}'")


def read_config(args: Optional[List[str]]) -> SimpleNamespace:
    """
    Parse command line arguments and return a JSON object.
    :returns: benchmark configuration.
    """
    parser = ArgumentParser()
    parser.add_argument("config", action=ActionJsonSchema(schema=SCHEMA), help="%s")
    config = parser.parse_args(args).config
    with mock.patch("jsonargparse.namespace.Namespace", SimpleNamespace):
        config = dict_to_namespace(config)

    # default num_warmup to half of num_sample
    if not hasattr(config, "num_warmup"):
        config.num_warmup = config.iterations // 2

    return config


def configure_logging(config: SimpleNamespace, output_dir: str) -> None:
    """
    Configure logging based on `config.loglevel` and add a stream handler.
    :param config: benchmark configuration
    :output_dir: directory to save the output
    """
    # set log level to INFO by default on root logger
    logging.getLogger().setLevel("INFO")
    # setup logging for all other requested loglevels
    if hasattr(config, "loglevels"):
        for key, val in config.loglevels.__dict__.items():
            logging.getLogger(key).setLevel(getattr(logging, val))
    # create a handler at the root level to display to stdout
    # and another to write to a log file
    for ch in [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(output_dir, "logging.txt"), encoding="utf-8"),
    ]:
        formatter = logging.Formatter(LOG_FORMAT)
        ch.setFormatter(formatter)
        logging.getLogger().addHandler(ch)
    LOGGER.debug(f"config - {str(config)}")
