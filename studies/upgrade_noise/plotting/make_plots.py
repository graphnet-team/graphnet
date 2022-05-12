import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


def add_truth(data, database):
    data = data.sort_values("event_no").reset_index(drop=True)
    with sqlite3.connect(database) as con:
        query = (
            "select event_no, energy, interaction_type, pid from truth where event_no in %s"
            % str(tuple(data["event_no"]))
        )
        truth = (
            pd.read_sql(query, con)
            .sort_values("event_no")
            .reset_index(drop=True)
        )

    truth["track"] = 0
    truth.loc[
        (abs(truth["pid"]) == 14) & (truth["interaction_type"] == 1), "track"
    ] = 1
    add_these = []
    for key in truth.columns:
        if key not in data.columns:
            add_these.append(key)
    for key in add_these:
        data[key] = truth[key]
    return data


def get_interaction_type(row):
    if row["interaction_type"] == 1:  # CC
        particle_type = "nu_" + {12: "e", 14: "mu", 16: "tau"}[abs(row["pid"])]
        return f"{particle_type} CC"
    else:
        return "NC"


def resolution_fn(r):
    if len(r) > 1:
        return (np.percentile(r, 84) - np.percentile(r, 16)) / 2.0
    else:
        return np.nan


def add_energylog10(df):
    df["energy_log10"] = np.log10(df["energy"])
    return df


def get_error(residual):
    rng = np.random.default_rng(42)
    w = []
    for i in range(150):
        new_sample = rng.choice(residual, size=len(residual), replace=True)
        w.append(resolution_fn(new_sample))
    return np.std(w)


def get_roc_and_auc(data, target):
    fpr, tpr, _ = roc_curve(data[target], data[target + "_pred"])
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def plot_roc(target, runids, save_dir, save_as_csv=False):
    width = 3.176 * 2
    height = 2.388 * 2
    fig = plt.figure(figsize=(width, height))
    for runid in runids:
        data = pd.read_csv(
            "/home/iwsatlas1/oersoe/phd/upgrade_noise/results/dev_step4_numu_%s_second_run/upgrade_%s_regression_45e_GraphSagePulses/results.csv"
            % (runid, target)
        )
        database = (
            "/mnt/scratch/rasmus_orsoe/databases/dev_step4_numu_%s_second_run/data/dev_step4_numu_%s_second_run.db"
            % (runid, runid)
        )
        if save_as_csv:
            data = add_truth(data, database)
            data = add_energylog10(data)
            data.to_csv(save_dir + "/%s_%s.csv" % (runid, target))
        # pulses_cut_val = 20
        # if runid == 140021:
        #    pulses_cut_val = 10
        fpr, tpr, auc = get_roc_and_auc(data, target)
        plt.plot(fpr, tpr, label=" %s : %s" % (runid, round(auc, 3)))
    plt.legend()
    plt.title("Track/Cascade Classification")
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.xlabel("False Positive Rate", fontsize=12)
    ymax = 0.3
    x_text = 0.2
    y_text = ymax - 0.05
    y_sep = 0.1
    plt.text(
        x_text,
        y_text - 0 * y_sep,
        "IceCubeUpgrade/nu_simulation/detector/step4/(%s,%s)"
        % (runids[0], runids[1]),
        va="top",
        fontsize=8,
    )
    plt.text(
        x_text,
        y_text - 1 * y_sep,
        "Pulsemaps used: SplitInIcePulses_GraphSage_Pulses ",
        va="top",
        fontsize=8,
    )
    plt.text(
        x_text,
        y_text - 2 * y_sep,
        "n_pulses > (%s, %s) selection applied during training" % (10, 20),
        va="top",
        fontsize=8,
    )

    fig.savefig(
        "/home/iwsatlas1/oersoe/phd/upgrade_noise/plots/preliminary_upgrade_performance_%s.pdf"
        % (target),
        bbox_inches="tight",
    )
    return


def calculate_width(data_sliced, target):
    track = data_sliced.loc[data_sliced["track"] == 1, :].reset_index(
        drop=True
    )
    cascade = data_sliced.loc[data_sliced["track"] == 0, :].reset_index(
        drop=True
    )
    if target == "energy":
        residual_track = (
            (track[target + "_pred"] - track[target]) / track[target]
        ) * 100
        residual_cascade = (
            (cascade[target + "_pred"] - cascade[target]) / cascade[target]
        ) * 100
    elif target == "zenith":
        residual_track = (track[target + "_pred"] - track[target]) * (
            360 / (2 * np.pi)
        )
        residual_cascade = (cascade[target + "_pred"] - cascade[target]) * (
            360 / (2 * np.pi)
        )
    else:
        residual_track = track[target + "_pred"] - track[target]
        residual_cascade = cascade[target + "_pred"] - cascade[target]

    return (
        resolution_fn(residual_track),
        resolution_fn(residual_cascade),
        get_error(residual_track),
        get_error(residual_cascade),
    )


def get_width(df, target):
    track_widths = []
    cascade_widths = []
    track_errors = []
    cascade_errors = []
    energy = []
    bins = np.arange(0, 3.1, 0.1)
    if target in ["zenith", "energy", "XYZ"]:
        for i in range(1, len(bins)):
            print(bins[i])
            idx = (df["energy_log10"] > bins[i - 1]) & (
                df["energy_log10"] < bins[i]
            )
            data_sliced = df.loc[idx, :].reset_index(drop=True)
            energy.append(np.mean(data_sliced["energy_log10"]))
            (
                track_width,
                cascade_width,
                track_error,
                cascade_error,
            ) = calculate_width(data_sliced, target)
            track_widths.append(track_width)
            cascade_widths.append(cascade_width)
            track_errors.append(track_error)
            cascade_errors.append(cascade_error)
        track_plot_data = pd.DataFrame(
            {
                "mean": energy,
                "width": track_widths,
                "width_error": track_errors,
            }
        )
        cascade_plot_data = pd.DataFrame(
            {
                "mean": energy,
                "width": cascade_widths,
                "width_error": cascade_errors,
            }
        )
        return track_plot_data, cascade_plot_data
    else:
        print("target not supported: %s" % target)


# Load data
def make_plot(target, runids, save_dir, save_as_csv=False):
    colors = {140021: "tab:blue", 140022: "tab:orange"}
    fig = plt.figure(constrained_layout=True)
    ax1 = plt.subplot2grid((6, 6), (0, 0), colspan=6, rowspan=6)
    for runid in runids:
        predictions_path = (
            "/home/iwsatlas1/oersoe/phd/upgrade_noise/results/dev_step4_numu_%s_second_run/upgrade_%s_regression_45e_GraphSagePulses/results.csv"
            % (runid, target)
        )
        database = (
            "/mnt/scratch/rasmus_orsoe/databases/dev_step4_numu_%s_second_run/data/dev_step4_numu_%s_second_run.db"
            % (runid, runid)
        )
        # pulses_cut_val = 20
        # if runid == 140021:
        #    pulses_cut_val = 10
        df = (
            pd.read_csv(predictions_path)
            .sort_values("event_no")
            .reset_index(drop=True)
        )
        df = add_truth(df, database)
        df = add_energylog10(df)
        if save_as_csv:
            df.to_csv(save_dir + "/%s_%s.csv" % (runid, target))
        plot_data_track, plot_data_cascade = get_width(df, target)

        ax1.plot(
            plot_data_track["mean"],
            plot_data_track["width"],
            linestyle="solid",
            lw=0.5,
            color="black",
            alpha=1,
        )
        ax1.fill_between(
            plot_data_track["mean"],
            plot_data_track["width"] - plot_data_track["width_error"],
            plot_data_track["width"] + plot_data_track["width_error"],
            color=colors[runid],
            alpha=0.8,
            label="Track %s" % runid,
        )

        ax1.plot(
            plot_data_cascade["mean"],
            plot_data_cascade["width"],
            linestyle="dashed",
            color="tab:blue",
            lw=0.5,
            alpha=1,
        )
        ax1.fill_between(
            plot_data_cascade["mean"],
            plot_data_cascade["width"] - plot_data_cascade["width_error"],
            plot_data_cascade["width"] + plot_data_cascade["width_error"],
            color=colors[runid],
            alpha=0.3,
            label="Cascade %s" % runid,
        )

        ax2 = ax1.twinx()
        ax2.hist(
            df["energy_log10"],
            histtype="step",
            label="deposited energy",
            color=colors[runid],
        )

    # plt.title('$\\nu_{v,u,e}$', size = 20)
    ax1.tick_params(axis="x", labelsize=6)
    ax1.tick_params(axis="y", labelsize=6)
    ax1.set_xlim((0, 3.1))

    leg = ax1.legend(frameon=False, fontsize=8)
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    if target == "energy":
        ax1.set_ylim((0, 175))
        ymax = 23.0
        y_sep = 8
        unit_tag = "(%)"
    else:
        unit_tag = "(deg.)"
    if target == "angular_res":
        target = "direction"
    if target == "XYZ":
        target = "vertex"
        unit_tag = "(m)"
    if target == "zenith":
        ymax = 10.0
        y_sep = 2.3
        ax1.set_ylim((0, 45))

    plt.tick_params(right=False, labelright=False)
    ax1.set_ylabel(
        "%s Resolution %s" % (target.capitalize(), unit_tag), size=10
    )
    ax1.set_xlabel("Energy  (log10 GeV)", size=10)

    x_text = 0.5
    y_text = ymax - 2.0
    ax1.text(
        x_text,
        y_text - 0 * y_sep,
        "IceCubeUpgrade/nu_simulation/detector/step4/(%s,%s)"
        % (runids[0], runids[1]),
        va="top",
        fontsize=8,
    )
    ax1.text(
        x_text,
        y_text - 1 * y_sep,
        "Pulsemaps used: SplitInIcePulses_GraphSage_Pulses ",
        va="top",
        fontsize=8,
    )
    ax1.text(
        x_text,
        y_text - 2 * y_sep,
        "n_pulses > (%s, %s) selection applied during training" % (10, 20),
        va="top",
        fontsize=8,
    )

    fig.suptitle("%s regression Upgrade MC using GNN" % target)

    # fig.suptitle('%s Resolution'%target.capitalize(), size = 12)
    fig.savefig(
        "/home/iwsatlas1/oersoe/phd/upgrade_noise/plots/preliminary_upgrade_performance_%s.pdf"
        % (target)
    )  # ,bbox_inches="tight")

    return


runids = [140021, 140022]
targets = ["zenith", "energy", "track"]
save_as_csv = True
save_dir = "/home/iwsatlas1/oersoe/phd/tmp/upgrade_csv"
for target in targets:
    if target != "track":
        make_plot(target, runids, save_dir, save_as_csv)
    else:
        plot_roc(target, runids, save_dir, save_as_csv)
