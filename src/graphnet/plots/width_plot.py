"""Function to make binned resolution plot."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graphnet.plots.utils import (
    calculate_statistics,
    calculate_relative_improvement_error,
    add_energy,
    add_pid_and_interaction,
)


def width_plot(
    key_limits,
    keys,
    key_bins,
    db,
    data_path,
    figsize=(10, 8),
    include_retro=True,
    track_cascade=True,
):
    """Makes a binned resolution plot.

    Will either divide into track/cascade or make a summary plot containing one curve for both topologies.

    Args:
        key_limits (dict): dictionary containing the key limits. Must have field called 'width'
        keys (list) : list of strings containing target variables. E.g. ['zenith', 'azimuth']
        key_bins (dict)  : dictionary containing the bins in which the data will be processed
        db (path): path to database containing RetroReco predictions and truth variables.
        data_path (path): path to csv containing the output of the model.
        figsize (tuple, optional): size of figure in inches. Defaults to (10,8).
        include_retro (bool, optional): include retro in plot?. Defaults to True.
        track_cascade (bool, optional): divide into track/cascade?. Defaults to True.

    Returns:
        matplotlib.pyplot.figure: figure
    """
    data = pd.read_csv(data_path)
    data = add_energy(db, data)
    data = add_pid_and_interaction(db, data)
    biases = calculate_statistics(data, keys, key_bins, db, include_retro)
    key_limits = key_limits["width"]
    if track_cascade is False:
        for key in biases["dynedge"].keys():
            fig = plt.figure(figsize=figsize)
            ax1 = plt.subplot2grid((6, 6), (0, 0), colspan=6, rowspan=4)
            ax2 = plt.subplot2grid((6, 6), (4, 0), colspan=6, rowspan=2)
            pid = "all_pid"
            interaction_type = str(1.0)
            plot_data = biases["dynedge"][key][pid][interaction_type]
            if include_retro:
                plot_data_retro = biases["retro"][key][pid][interaction_type]
            if len(plot_data["mean"]) != 0:
                ax3 = ax1.twinx()
                ax3.bar(
                    x=(plot_data["mean"]),
                    height=plot_data["count"],
                    alpha=0.3,
                    color="grey",
                    align="center",
                    width=0.25,
                )
                ax1.errorbar(
                    plot_data["mean"],
                    plot_data["width"],
                    plot_data["width_error"],
                    linestyle="dotted",
                    fmt="o",
                    capsize=10,
                    label="dynedge",
                )
                if include_retro:
                    ax1.errorbar(
                        plot_data_retro["mean"],
                        plot_data_retro["width"],
                        plot_data_retro["width_error"],
                        linestyle="dotted",
                        fmt="o",
                        capsize=10,
                        label="RetroReco",
                    )
                labels = [item.get_text() for item in ax1.get_xticklabels()]
                empty_string_labels = [""] * len(labels)
                ax1.set_xticklabels(empty_string_labels)
                ax1.grid()
                ax2.plot(
                    plot_data["mean"],
                    np.repeat(0, len(plot_data["mean"])),
                    color="black",
                    lw=2,
                )
                if include_retro:
                    ax2.errorbar(
                        plot_data["mean"],
                        1
                        - np.array(plot_data["width"])
                        / np.array(plot_data_retro["width"]),
                        calculate_relative_improvement_error(
                            1
                            - np.array(plot_data["width"])
                            / np.array(plot_data_retro["width"]),
                            plot_data["width"],
                            plot_data["width_error"],
                            plot_data_retro["width"],
                            plot_data_retro["width_error"],
                        ),
                        marker="o",
                        capsize=10,
                        markeredgecolor="black",
                    )

                # plt.title('$\\nu_{v,u,e}$', size = 20)
                ax1.tick_params(axis="x", labelsize=10)
                ax1.tick_params(axis="y", labelsize=10)
                ax1.set_xlim(key_limits[key]["x"])
                ax2.set_xlim(key_limits[key]["x"])
                ax2.set_ylim([-0.1, 0.55])
                ax1.legend()
                if key == "energy":
                    unit_tag = "[%]"
                else:
                    unit_tag = "[deg.]"
                plt.tick_params(right=False, labelright=False)
                ax1.set_ylabel("%s Resolution %s" % (key, unit_tag), size=15)
                ax2.set_xlabel("$Energy_{log10}$ [GeV]", size=15)
                ax2.set_ylabel("Rel. Impro.", size=15)

                fig.suptitle("%s CC" % key, size=20)
                # fig.savefig('performance_%s.png'%key)
    else:
        for key in biases["dynedge"].keys():
            fig = plt.figure(figsize=figsize)
            ax1 = plt.subplot2grid((6, 6), (0, 0), colspan=6, rowspan=4)
            ax2 = plt.subplot2grid((6, 6), (4, 0), colspan=6, rowspan=2)
            pid = "track"
            interaction_type = str(1.0)
            plot_data_track = biases["dynedge"][key][str(14.0)][str(1.0)]
            plot_data_cascade = biases["dynedge"][key]["cascade"]
            if include_retro:
                plot_data_retro_track = biases["retro"][key][str(14.0)][
                    str(1.0)
                ]
                plot_data_retro_cascade = biases["retro"][key]["cascade"]
            if len(plot_data_track["mean"]) != 0:
                ax3 = ax1.twinx()
                # ax3.bar(x = plot_data_track['mean'], height = plot_data_track['count'],
                #        alpha = 0.3,
                #        color = 'grey',
                #        align = 'center',
                #        width = 0.25)
                ax1.errorbar(
                    plot_data_track["mean"],
                    plot_data_track["width"],
                    plot_data_track["width_error"],
                    linestyle="dotted",
                    fmt="o",
                    capsize=10,
                    color="blue",
                    label="GCN-all Track",
                )
                ax1.errorbar(
                    plot_data_cascade["mean"],
                    plot_data_cascade["width"],
                    plot_data_cascade["width_error"],
                    linestyle="solid",
                    fmt="o",
                    capsize=10,
                    color="darkblue",
                    label="GCN-all Cascade",
                )
                if include_retro:
                    ax1.errorbar(
                        plot_data_retro_track["mean"],
                        plot_data_retro_track["width"],
                        plot_data_retro_track["width_error"],
                        linestyle="dotted",
                        fmt="o",
                        capsize=10,
                        color="orange",
                        label="RetroReco Track",
                    )
                    ax1.errorbar(
                        plot_data_retro_cascade["mean"],
                        plot_data_retro_cascade["width"],
                        plot_data_retro_cascade["width_error"],
                        linestyle="solid",
                        fmt="o",
                        capsize=10,
                        color="darkorange",
                        label="RetroReco Cascade",
                    )
                labels = [item.get_text() for item in ax1.get_xticklabels()]
                empty_string_labels = [""] * len(labels)
                ax1.set_xticklabels(empty_string_labels)
                ax1.grid()
                ax2.plot(
                    plot_data_track["mean"],
                    np.repeat(0, len(plot_data_track["mean"])),
                    color="black",
                    lw=2,
                )
                if include_retro:
                    ax2.errorbar(
                        plot_data_track["mean"],
                        1
                        - np.array(plot_data_track["width"])
                        / np.array(plot_data_retro_track["width"]),
                        calculate_relative_improvement_error(
                            1
                            - np.array(plot_data_track["width"])
                            / np.array(plot_data_retro_track["width"]),
                            plot_data_track["width"],
                            plot_data_track["width_error"],
                            plot_data_retro_track["width"],
                            plot_data_retro_track["width_error"],
                        ),
                        marker="o",
                        capsize=10,
                        markeredgecolor="black",
                        color="limegreen",
                        label="track",
                        linestyle="dotted",
                    )
                    ax2.errorbar(
                        plot_data_cascade["mean"],
                        1
                        - np.array(plot_data_cascade["width"])
                        / np.array(plot_data_retro_cascade["width"]),
                        calculate_relative_improvement_error(
                            1
                            - np.array(plot_data_cascade["width"])
                            / np.array(plot_data_retro_cascade["width"]),
                            plot_data_cascade["width"],
                            plot_data_cascade["width_error"],
                            plot_data_retro_cascade["width"],
                            plot_data_retro_cascade["width_error"],
                        ),
                        marker="o",
                        capsize=10,
                        markeredgecolor="black",
                        color="springgreen",
                        label="cascade",
                        linestyle="solid",
                    )
                    ax2.legend()
                # plt.title('$\\nu_{v,u,e}$', size = 20)
                ax1.tick_params(axis="x", labelsize=10)
                ax1.tick_params(axis="y", labelsize=10)
                ax1.set_xlim(key_limits[key]["x"])
                ax2.set_xlim(key_limits[key]["x"])
                ax2.set_ylim([-0.40, 0.40])
                ax1.legend()
                if key == "energy":
                    unit_tag = "[%]"
                else:
                    unit_tag = "[deg.]"
                plt.tick_params(right=False, labelright=False)
                ax1.set_ylabel("%s Resolution %s" % (key, unit_tag), size=15)
                ax2.set_xlabel("$Energy_{log10}$ [GeV]", size=15)
                ax2.set_ylabel("Rel. Impro.", size=15)

                fig.suptitle("%s Performance" % key, size=20)
                # fig.savefig('performance_track_cascade_%s.png'%key)

    return fig
