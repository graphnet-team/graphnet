import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3

#from graphnet.plots.width_plot import width_plot

# Load data
predictions_path = '//groups/icecube/asogaard/gnn/results/upgrade_test_1/dev_upgrade_step4_preselection_decemberv2/test_upgrade_zenith_regression/results.csv'
database = '/groups/icecube/asogaard/data/sqlite/dev_upgrade_step4_preselection_decemberv2/data/dev_upgrade_step4_preselection_decemberv2.db'

df_pred = pd.read_csv(predictions_path, index_col=0).astype({'event_no': int, 'n_pulses': int}).sort_values('event_no')

event_no_string = '({})'.format(', '.join(map(str, df_pred.event_no.values)))
with sqlite3.connect(database) as conn:
    df_truth = pd.read_sql(f"SELECT event_no, pid, interaction_type, zenith, energy FROM truth WHERE event_no IN {event_no_string}", conn)

df_truth = df_truth.astype({'event_no': int, 'pid': int, 'interaction_type': int, 'zenith': float, 'energy': float})

df = df_pred.merge(df_truth, on="event_no", how="outer")

# Prepare data
def track_or_cascade(row):
    track = (abs(row['pid']) == 14) & (row['interaction_type'] == 1)
    cascade = (abs(row['pid']) != 14) | (row['interaction_type'] != 1)
    if track:
        return "track"
    elif cascade:
        return "cascade"
    return "N/A"

df = df.assign(type=df.apply(track_or_cascade, axis=1))
assert (df['zenith_x'] - df['zenith_y']).abs().max() < 1e-06
assert (df['energy_x'] - df['energy_y']).abs().max() < 1e-03

df = df[['zenith_x', 'zenith_pred', 'energy_x', 'n_pulses', 'type']]
df = df.rename(columns={'zenith_x': 'zenith', 'energy_x': 'energy'})
df[['zenith', 'zenith_pred']] *= 180. / np.pi
df = df.assign(residual=df.zenith_pred - df.zenith)

# Make plot
xmin, xmax = -1, 4
bins = np.linspace(xmin, xmax, 5 * 2 + 1)
bin_centers = bins[:-1] + np.diff(bins) * 0.5
ix_bin = np.digitize(np.clip(np.log10(df['energy']), xmin, xmax), bins) - 1
df = df.assign(ix_bin=ix_bin)

def resolution_fn(r):
    return (np.percentile(r, 84) - np.percentile(r, 16)) / 2.

fig, ax = plt.subplots(figsize=(10,8))


plt.plot(np.nan, np.nan, lw=2, color='b', label=f'Track')
plt.plot(np.nan, np.nan, lw=2, color='r', label=f'Cascade')

min_samples = 150
for ix, min_pulses in enumerate([0, 15, 30]):
    # Selection formatting
    lw = 2  # 1 + ix / 2.
    ls = {
        0: ':',
        #1: '-.',
        1: '--',
        2: '-',
    }[ix]
    marker = {
        0: 's',
        1: '^',
        2: 'o',
    }[ix]

    # Selection
    df_selected = df.query(f"n_pulses >= {min_pulses} ")
    df_plot = df_selected.groupby(['type', 'ix_bin']).agg({'residual': resolution_fn}).rename(columns={'residuals': 'resolution'})

    # Bootstrap sampling for estimating uncertainty on resolution
    realisations = []
    for _ in range(100):
        df_realisation = df_selected.sample(frac=1., replace=True).groupby(['type', 'ix_bin']).agg({'residual': resolution_fn}).rename(columns={'residuals': 'resolution'})
        realisations.append(df_realisation)

    df_realisations = pd.concat(realisations, axis=1)
    uncert = df_realisations.std(axis=1)

    # Masking out configuration with too few samples
    mask = df_selected.groupby(['type', 'ix_bin']).count().mean(axis=1) >= min_samples
    df_plot = df_plot[mask]
    uncert = uncert[mask]

    # Plotting mean resolution
    plt.plot(bin_centers[df_plot.loc['track'].index],   df_plot.loc['track'].values,   color='b', lw=lw, ls=ls, marker=marker)
    plt.plot(bin_centers[df_plot.loc['cascade'].index], df_plot.loc['cascade'].values, color='r', lw=lw, ls=ls, marker=marker)

    # Plotting estimated uncertainty
    plt.fill_between(bin_centers[df_plot.loc['track'].index],   (df_plot.loc['track']  .mean(axis=1) - uncert.loc['track'])  .values.ravel(), (df_plot.loc['track']  .mean(axis=1) + uncert.loc['track'])  .values.ravel(), color='b', alpha=0.3)
    plt.fill_between(bin_centers[df_plot.loc['cascade'].index], (df_plot.loc['cascade'].mean(axis=1) - uncert.loc['cascade']).values.ravel(), (df_plot.loc['cascade'].mean(axis=1) + uncert.loc['cascade']).values.ravel(), color='r', alpha=0.3)

    # Legend entry for selection
    selection_label = "$n_{pulses} \geq %d$" % min_pulses
    plt.plot(np.nan, np.nan, color='gray', lw=lw, ls=ls, marker=marker, label=selection_label)

ymax = 55.
plt.ylabel('Zenith resolution (68% IPR) [deg.]')
plt.xlabel('log10(energy) [log10(GeV)]')
x_text = -0.9
y_text = ymax - 2.
y_sep = 2.3
plt.text(x_text, y_text - 0 * y_sep, "IceCubeUpgrade/nu_simulation/detector/step4", va='top')
plt.text(x_text, y_text - 1 * y_sep, "Pulsemaps used:\n    IceCubePulsesTWSRT\n    I3RecoPulseSeriesMapRFCleaned_mDOM\n    I3RecoPulseSeriesMapRFCleaned_DEgg", va='top')

x_text = 1.4
y_text = ymax - 2.
plt.text(x_text, y_text - 0 * y_sep, "Trained on equal-flavour mix (3 x 286K events)", va='top')
plt.text(x_text, y_text - 1 * y_sep, "No selection applied", va='top')
plt.text(x_text, y_text - 2 * y_sep, "asogaard/graphnet:training-on-upgrade-mc@9d5207e5", va='top')
plt.suptitle("Neutrino zenith regression in IceCube Upgrade MC using GNNs (07/01/22)")
plt.ylim(0, ymax)
plt.xlim(-1,4)
fig.tight_layout()

# Shrink current axis
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.86, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('220107_preliminary_upgrade_performance_zenith.png')

'''
emin, emax = -1, 4
estep = 0.5
key_limits = {'bias':{'energy':{'x':[emin, emax], 'y':[-100,100]},
                        'zenith': {'x':[emin, emax], 'y':[-100,100]}},
            'width':{'energy':{'x':[emin, emax], 'y':[-0.5,1.5]},
                        'zenith': {'x':[emin, emax], 'y':[-100,100]}},
            'rel_imp':{'energy':{'x':[emin, emax], 'y':[-0.75,0.75]}},
            'osc':{'energy':{'x':[emin, emax], 'y':[-0.75,0.75]}},
            'distributions':{'energy':{'x':[emin, emax], 'y':[-0.75,0.75]}}}
keys = ['zenith']
key_bins = { 'energy': np.arange(emin, emax + estep, estep),
            'zenith': np.arange(0, 180, 10) }

performance_figure = width_plot(key_limits, keys, key_bins, database, predictions_path, figsize = (10,8), include_retro = False, track_cascade = True,
    #filter_query="n_pulses > 40",
)

performance_figure.savefig('test_upgrade_performance.png')
'''