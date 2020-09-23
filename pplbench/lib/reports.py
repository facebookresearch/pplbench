# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
from types import SimpleNamespace
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xarray as xr

from .ppl_helper import PPLDetails


def generate_plots(
    output_dir: str,
    config: SimpleNamespace,
    all_ppl_details: List[PPLDetails],
    all_variable_metrics_data: xr.Dataset,
    all_other_metrics_data: xr.Dataset,
) -> None:
    if hasattr(config, "figures") and not getattr(config.figures, "generate_pll", True):
        return
    # we will plot PLL for varying ranges of samples to give a better picture
    generate_pll_plot(
        config, output_dir, all_ppl_details, all_other_metrics_data.pll, "pll"
    )
    num_samples = len(all_other_metrics_data.pll.coords["draw"])
    generate_pll_plot(
        config,
        output_dir,
        all_ppl_details,
        all_other_metrics_data.pll.isel(draw=slice(num_samples // 4, None)),
        "pll_three_quarter",
    )
    generate_pll_plot(
        config,
        output_dir,
        all_ppl_details,
        all_other_metrics_data.pll.isel(draw=slice(num_samples // 2, None)),
        "pll_half",
    )
    generate_pll_plot(
        config,
        output_dir,
        all_ppl_details,
        all_other_metrics_data.pll.isel(draw=slice(3 * num_samples // 4, None)),
        "pll_quarter",
    )


def generate_pll_plot(
    config: SimpleNamespace,
    output_dir: str,
    all_ppl_details: List[PPLDetails],
    pll: xr.DataArray,
    file_prefix: str,
) -> None:
    suffix = (
        config.figures.suffix
        if hasattr(config, "figures") and hasattr(config.figures, "suffix")
        else "png"
    )
    min_pll = pll.min("chain")
    max_pll = pll.max("chain")
    mean_pll = pll.mean("chain")
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({"font.size": 18})

    legend = []
    for ppl_details in all_ppl_details:
        (line,) = plt.plot(
            pll.coords["draw"],
            mean_pll.sel(ppl=ppl_details.name),
            color=ppl_details.color,
            label=ppl_details.name,
        )
        plt.fill_between(
            pll.coords["draw"],
            min_pll.sel(ppl=ppl_details.name),
            max_pll.sel(ppl=ppl_details.name),
            color=ppl_details.color,
            interpolate=True,
            alpha=0.3,
        )
        legend.append(line)
    legend = sorted(legend, key=lambda line: line.get_label())
    plt.legend(handles=legend, loc="lower right")
    ax = plt.gca()
    ax.set_xlabel("Samples")
    ax.set_ylabel("Predictive Log Likelihood")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3e"))
    plt.savefig(
        os.path.join(output_dir, f"{file_prefix}.{suffix}"),
        bbox_inches="tight",
        dpi=300,
    )
