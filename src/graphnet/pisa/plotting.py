"""Functions for plotting contours from PISA fits."""

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


def read_entry(entry: Dict) -> Tuple[Any, ...]:
    """Parse the contents of `entry`."""
    path = entry["path"]
    model_name = entry["model"]
    try:
        label = entry["label"]
    except KeyError:
        label = path.split("/")[-2]
    try:
        ls = entry["linestyle"]
    except KeyError:
        ls = "-"
    try:
        color = entry["color"]
    except KeyError:
        color = None
    entry_data = pd.read_csv(path)

    return entry_data, model_name, label, ls, color


def plot_2D_contour(
    contour_data: List[Dict],
    xlim: Tuple[float, float] = (0.4, 0.6),
    ylim: Tuple[float, float] = (2.38 * 1e-3, 2.55 * 1e-3),
    chi2_critical_value: float = 4.605,
    width: float = 3.176,
    height: float = 2.388,
) -> Figure:
    """Plot 2D contours from GraphNeT PISA fits.

    Args:
        contour_data: List of dictionaries with plotting information. Format is
            for each dictionary is:
                {'path': path_to_pisa_fit_result,
                 'model': 'name_of_my_model_in_fit'}.
            One can specify optional fields in the dictionary: "label" - the
            legend label, "color" - the color of the contour, "linestyle" - the
            style of the contour line.
        xlim: Lower and upper bound of x-axis.
        ylim: Lower and upper bound of y-axis.
        chi2_critical_value: The critical value of the chi2 fits. Defaults to
            4.605 (90% CL). @NOTE: This, and the below, can't both be right.
        width: width of figure in inches.
        height: height of figure in inches.

    Returns:
        The figure with contours.
    """
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    proxy = []
    labels = []
    for entry in contour_data:
        entry_data, model_name, label, ls, color = read_entry(entry)
        model_idx = entry_data["model"] == model_name
        model_data = entry_data.loc[model_idx]
        x = pd.unique(model_data.sort_values("theta23_fixed")["theta23_fixed"])
        y = pd.unique(model_data.sort_values("dm31_fixed")["dm31_fixed"])
        z = np.zeros((len(y), len(x)))
        for i in range(len(x)):
            for k in range(len(y)):
                idx = (model_data["theta23_fixed"] == x[i]) & (
                    model_data["dm31_fixed"] == y[k]
                )
                match = model_data["mod_chi2"][idx]
                if len(match) > 0:
                    if model_data["converged"][idx].values is True:
                        match = float(match)
                    else:
                        match = 10000  # Sets the z value very high to exclude it from contour
                else:
                    match = 10000  # Sets the z value very high to exclude it from contour
                z[k, i] = match

        CS = ax.contour(
            np.sin(np.deg2rad(x)) ** 2,
            y,
            z,
            levels=[chi2_critical_value],
            colors=color,
            label=label,
            linestyles=ls,
            linewidths=2,
        )
        # ax.clabel(CS, inline=1, fontsize=10)
        proxy.extend(
            [plt.Rectangle((0, 0), 1, 1, fc=color) for pc in CS.collections]
        )
        if chi2_critical_value == 4.605:
            label = label + " 90 $\\%$ CL"
        labels.append(label)
    plt.legend(proxy, labels, frameon=False, loc="upper right")
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel("$\\sin^2(\\theta_{23})$", fontsize=12)
    plt.ylabel("$\\Delta m_{31}^2 [eV^2]$", fontsize=12)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Sensitivity (Simplified Analysis)")
    return fig


def plot_1D_contour(
    contour_data: List[Dict],
    chi2_critical_value: float = 2.706,
    width: float = 2 * 3.176,
    height: float = 2.388,
) -> Figure:
    """Plot 1D contours from GraphNeT PISA fits.

    Args:
        contour_data: List of dictionaries with plotting information. Format is
            for each dictionary is:
                {'path': path_to_pisa_fit_result,
                 'model': 'name_of_my_model_in_fit'}.
            One can specify optional fields in the dictionary: "label" - the
            legend label, "color" - the color of the contour, "linestyle" - the
            style of the contour line.
        chi2_critical_value: The critical value of the chi2 fits. Defaults to
            2.706 (90% CL). @NOTE: This, and the above, can't both be right.
        width: width of figure in inches.
        height: height of figure in inches.

    Returns:
        The figure with contours.
    """
    variables = ["theta23_fixed", "dm31_fixed"]
    fig, ax = plt.subplots(
        1, 2, figsize=(width, height), constrained_layout=True
    )
    ls = 0
    for entry in contour_data:
        entry_data, model_name, label, ls, color = read_entry(entry)
        for variable in variables:
            model_idx = entry_data["model"] == model_name
            padding_idx = entry_data[variable] != -1
            chi2_idx = entry_data["mod_chi2"] < chi2_critical_value
            model_data = entry_data.loc[
                (model_idx) & (padding_idx) & (chi2_idx), :
            ]
            x = model_data.sort_values(variable)
            if variable == "theta23_fixed":
                ax[0].set_ylabel("$\\chi^2$", fontsize=12)
                ax[0].plot(
                    np.sin(np.deg2rad(x[variable])) ** 2,
                    x["mod_chi2"],
                    color=color,
                    label=label,
                    ls=ls,
                )
                ax[0].set_xlabel("$\\sin(\\theta_{23})^2$", fontsize=12)
            elif variable == "dm31_fixed":
                ax[1].plot(
                    x[variable], x["mod_chi2"], color=color, label=label, ls=ls
                )
                ax[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                ax[1].set_xlabel("$\\Delta m_{31}^2 [eV^2]$", fontsize=12)
    h = [item.get_text() for item in ax[1].get_yticklabels()]
    empty_string_labels = [""] * len(h)
    ax[1].set_yticklabels(empty_string_labels)
    ax[0].set_ylim(0, chi2_critical_value)
    ax[1].set_ylim(0, chi2_critical_value)
    ax[0].legend()
    return fig
