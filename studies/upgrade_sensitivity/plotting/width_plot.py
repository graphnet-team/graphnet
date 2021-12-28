from graphnet.plots.width_plot import width_plot
import numpy as np


predictions_path = '/groups/icecube/asogaard/gnn/results/dev_upgrade_step4_preselection_decemberv2/test_upgrade_mc_mdom_zenith_regression/results.csv'
database    = '/groups/icecube/asogaard/data/sqlite/dev_upgrade_step4_preselection_decemberv2/data/dev_upgrade_step4_preselection_decemberv2.db'

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

performance_figure = width_plot(key_limits, keys, key_bins, database, predictions_path, figsize = (10,8), include_retro = False, track_cascade = True)

performance_figure.savefig('test_performance_figures.png')