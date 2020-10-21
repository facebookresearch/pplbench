# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import hashlib
import logging
import struct
import time
from types import SimpleNamespace
from typing import Dict, List, NamedTuple, Tuple, Type

import arviz
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import is_color_like, to_rgb

from ..models.base_model import BaseModel
from ..ppls.base_ppl_impl import BasePPLImplementation
from ..ppls.base_ppl_inference import BasePPLInference
from .utils import load_class_or_exit, save_dataset


class PPLDetails(NamedTuple):
    name: str
    seed: int
    color: Tuple[float, float, float]
    impl_class: Type[BasePPLImplementation]
    inference_class: Type[BasePPLInference]
    compile_args: Dict
    infer_args: Dict


LOGGER = logging.getLogger("pplbench")


def find_ppl_details(config: SimpleNamespace) -> List[PPLDetails]:
    """
    Returns information about each instance of PPL inference that is requested
    in the benchmark.
    This raises a Runtime exception if the names are not unique.
    :param config: The benchmark configuration object.
    :returns: A list of information objects one per ppl inference.
    """
    ret_val = []
    prev_names = set()
    model_class = getattr(config.model, "class")
    for ppl_config in config.ppls:
        package = getattr(ppl_config, "package", "pplbench.ppls." + ppl_config.name)
        impl_class = load_class_or_exit(f"{package}.{model_class}")
        inference_class = load_class_or_exit(
            f"{package}.{getattr(ppl_config.inference, 'class')}"
        )
        # create a unique name for the PPL inference if not given
        name = (
            ppl_config.legend.name
            if hasattr(ppl_config, "legend") and hasattr(ppl_config.legend, "name")
            else ppl_config.name + "-" + inference_class.__name__
        )
        if name in prev_names:
            raise RuntimeError(f"duplicate PPL inference {name}")
        prev_names.add(name)

        # we will generate a unique color for each ppl name
        # https://stackoverflow.com/questions/40351791/how-to-hash-strings-into-a-float-in-01
        def _hash(name):
            return (
                float(
                    struct.unpack(
                        "L", hashlib.sha256(bytes(name, "utf-8")).digest()[:8]
                    )[0]
                )
                / 2 ** 64
            )

        def _get_color(name):
            return (_hash(name + "0"), _hash(name + "1"), _hash(name + "1"))

        if hasattr(ppl_config, "legend") and hasattr(ppl_config.legend, "color"):
            if is_color_like(ppl_config.legend.color):
                color = to_rgb(ppl_config.legend.color)
            else:
                raise RuntimeError(
                    f"invalid color '{ppl_config.legend.color}' for PPL inference '{name}'"
                )
        else:
            color = _get_color(name)
        # finally pick a default seed for the ppl
        seed = getattr(ppl_config, "seed", int(time.time() + 19))
        infer = ppl_config.inference
        ret_val.append(
            PPLDetails(
                name=name,
                seed=seed,
                color=color,
                impl_class=impl_class,
                inference_class=inference_class,
                compile_args=infer.compile_args.__dict__
                if hasattr(infer, "compile_args")
                else {},
                infer_args=infer.infer_args.__dict__
                if hasattr(infer, "infer_args")
                else {},
            )
        )
        LOGGER.debug(f"added PPL inference '{str(ret_val[-1])}'")
    return ret_val


def collect_samples_and_stats(
    config: SimpleNamespace,
    model_cls: Type[BaseModel],
    all_ppl_details: List[PPLDetails],
    train_data: xr.Dataset,
    test_data: xr.Dataset,
    output_dir: str,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    :param confg: The benchmark configuration.
    :param model_cls: The model class
    :param ppl_details: For each ppl the the impl and inference classes etc.
    :param train_data: The training dataset.
    :param test_data: The held-out test dataset.
    :param output_dir: The directory for storing results.
    :returns: Two datasets:
        variable_metrics
            Coordinates: ppl, metric (n_eff, Rhat), others from model
            Data variables: from model
        other_metrics
            Coordinates: ppl, chain, draw, phase (compile, infer)
            Data variables: pll (ppl, chain, draw), timing (ppl, chain, phase)
    """
    all_variable_metrics, all_pll, all_timing, all_names = [], [], [], []
    all_samples, all_overall_neff, all_overall_neff_per_time = [], [], []
    for pplobj in all_ppl_details:
        all_names.append(pplobj.name)
        rand = np.random.RandomState(pplobj.seed)
        LOGGER.info(f"Starting inference on `{pplobj.name}` with seed {pplobj.seed}")
        # first compile the PPL Implementation this involves two steps
        compile_t1 = time.time()
        # compile step 1: instantiate ppl inference object
        infer_obj = pplobj.inference_class(pplobj.impl_class, train_data.attrs)
        # compile step 2: call compile
        infer_obj.compile(seed=rand.randint(1, 1e7), **pplobj.compile_args)
        compile_time = time.time() - compile_t1
        LOGGER.info(f"compiling on `{pplobj.name}` took {compile_time:.2f} secs")
        # then run inference for each trial
        trial_samples, trial_pll, trial_timing = [], [], []
        for trialnum in range(config.trials):
            infer_t1 = time.time()
            samples = infer_obj.infer(
                data=train_data,
                num_samples=config.num_samples,
                seed=rand.randint(1, 1e7),
                **pplobj.infer_args,
            )
            infer_time = time.time() - infer_t1
            LOGGER.info(f"inference trial {trialnum} took {infer_time:.2f} secs")
            # compute the pll per sample and then convert it to the actual pll over
            # cumulative samples
            persample_pll = model_cls.evaluate_posterior_predictive(samples, test_data)
            pll = np.logaddexp.accumulate(persample_pll) - np.log(
                np.arange(config.num_samples) + 1
            )
            LOGGER.info(f"PLL = {str(pll)}")
            trial_samples.append(samples)
            trial_pll.append(pll)
            trial_timing.append([compile_time, infer_time])
            # finally, give the inference object an opportunity
            # to write additional diagnostics
            infer_obj.additional_diagnostics(output_dir, f"{pplobj.name}_{trialnum}")
        del infer_obj
        # concatenate the samples data from each trial together so we can compute metrics
        trial_samples_data = xr.concat(
            trial_samples, pd.Index(data=np.arange(config.trials), name="chain")
        )
        neff_data = arviz.ess(trial_samples_data)
        rhat_data = arviz.rhat(trial_samples_data)
        LOGGER.info(f"Trials completed for {pplobj.name}")
        LOGGER.info("== n_eff ===")
        LOGGER.info(str(neff_data.data_vars))
        LOGGER.info("==  Rhat ===")
        LOGGER.info(str(rhat_data.data_vars))

        # compute ess/time
        neff_df = neff_data.to_dataframe()
        overall_neff = [
            neff_df.values.min(),
            np.median(neff_df.values),
            neff_df.values.max(),
        ]
        mean_inference_time = np.mean(np.array(trial_timing)[:, 1])
        overall_neff_per_time = np.array(overall_neff) / mean_inference_time

        LOGGER.info("== overall n_eff [min, median, max]===")
        LOGGER.info(str(overall_neff))
        LOGGER.info("== overall n_eff/s [min, median, max]===")
        LOGGER.info(str(overall_neff_per_time))

        trial_variable_metrics_data = xr.concat(
            [neff_data, rhat_data], pd.Index(data=["n_eff", "Rhat"], name="metric")
        )
        all_variable_metrics.append(trial_variable_metrics_data)
        all_pll.append(trial_pll)
        all_timing.append(trial_timing)
        all_samples.append(trial_samples_data)
        all_overall_neff.append(overall_neff)
        all_overall_neff_per_time.append(overall_neff_per_time)
    # merge the trial-level metrics at the PPL level
    all_variable_metrics_data = xr.concat(
        all_variable_metrics, pd.Index(data=all_names, name="ppl")
    )
    all_other_metrics_data = xr.Dataset(
        {
            "timing": (["ppl", "chain", "phase"], all_timing),
            "pll": (["ppl", "chain", "draw"], all_pll),
            "overall_neff": (["ppl", "percentile"], all_overall_neff),
            "overall_neff_per_time": (["ppl", "percentile"], all_overall_neff_per_time),
        },
        coords={
            "ppl": np.array(all_names),
            "chain": np.arange(config.trials),
            "phase": np.array(["compile", "infer"]),
            "draw": np.arange(config.num_samples),
            "percentile": np.array(["min", "median", "max"]),
        },
    )
    all_samples_data = xr.concat(all_samples, pd.Index(data=all_names, name="ppl"))
    model_cls.additional_metrics(output_dir, all_samples_data, train_data, test_data)
    LOGGER.info("all benchmark samples and metrics collected")
    # save the samples data only if requested
    if getattr(config, "save_samples", False):
        save_dataset(output_dir, "samples", all_samples_data)
    # write out thes metrics
    save_dataset(output_dir, "diagnostics", all_variable_metrics_data)
    save_dataset(output_dir, "metrics", all_other_metrics_data)
    return all_variable_metrics_data, all_other_metrics_data
