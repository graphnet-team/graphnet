import numpy as np
from uncertainties import unumpy as unp
import pisa
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.pipeline import Pipeline
from pisa.analysis.analysis import Analysis
from pisa import FTYPE, ureg
import pandas as pd
import multiprocessing
import os
import random


def make_binning_cfg(config_dict, outdir):
    template = """graphnet_dynamic_binning.order = reco_energy, reco_coszen, pid
graphnet_dynamic_binning.reco_energy = {'num_bins':%s, 'is_log':True, 'domain':[0.5,55] * units.GeV, 'tex': r'E_{\\rm reco}'}
graphnet_dynamic_binning.reco_coszen = {'num_bins':%s, 'is_lin':True, 'domain':[-1,1], 'tex':r'\\cos{\\theta}_{\\rm reco}'}
graphnet_dynamic_binning.pid = {'bin_edges': %s, 'tex':r'{\\rm PID}'}

true_allsky_fine.order = true_energy, true_coszen
true_allsky_fine.true_energy = {'num_bins':%s, 'is_log':True, 'domain':[1,1000] * units.GeV, 'tex': r'E_{\\rm true}'}
true_allsky_fine.true_coszen = {'num_bins':%s, 'is_lin':True, 'domain':[-1,1], 'tex':r'\\cos\,\\theta_{Z,{\\rm true}}'}
"""%(config_dict['reco_energy']['num_bins'], 
        config_dict['reco_coszen']['num_bins'], 
        config_dict['pid']['bin_edges'], 
        config_dict['true_energy']['num_bins'],
        config_dict['true_coszen']['num_bins'])

    if config_dict['post_fix'] != None:
        filename = 'binning_config_%s.cfg'%config_dict['post_fix']
        with open(outdir + '/' + filename, 'w+') as f:
            f.write(template)
    else:
        filename = 'binning_config.cfg'
        with open(outdir + '/' + filename, 'w+') as f:
            f.write(template)
    return outdir + '/' + filename

def make_pipeline_cfg(config_dict, outdir):
    template = """#include %s as binning
#include settings/osc/nufitv20.cfg as osc
#include settings/osc/earth.cfg as earth

[pipeline]
order = data.sqlite_loader, flux.honda_ip, flux.barr_simple, osc.prob3, aeff.aeff, utils.hist
param_selections = nh
name = neutrinos
output_binning = graphnet_dynamic_binning
output_key = weights, errors

[data.sqlite_loader]
calc_mode = events
apply_mode = events
output_names = nue_cc, numu_cc, nutau_cc, nue_nc, numu_nc, nutau_nc, nuebar_cc, numubar_cc, nutaubar_cc, nuebar_nc, numubar_nc, nutaubar_nc
post_fix = %s
database = %s

[flux.honda_ip]
calc_mode = true_allsky_fine
apply_mode = events
param.flux_table = flux/honda-2015-spl-solmin-aa.d

[flux.barr_simple]
calc_mode = true_allsky_fine
apply_mode = events
param.nu_nubar_ratio = 1.0 +/- 0.1
param.nu_nubar_ratio.fixed = True
param.nu_nubar_ratio.range = nominal + [-3., +3.] * sigma
param.nue_numu_ratio = 1.0 +/- 0.05
param.nue_numu_ratio.fixed = False
param.nue_numu_ratio.range = nominal + [-0.5, +0.5]
param.Barr_uphor_ratio = 0.0 +/- 1.0
param.Barr_uphor_ratio.fixed = False
param.Barr_uphor_ratio.range = nominal + [-3.0, +3.0]
param.Barr_nu_nubar_ratio = 0.0 +/- 1.0
param.Barr_nu_nubar_ratio.fixed = False
param.Barr_nu_nubar_ratio.range = nominal + [-3.0, +3.0]
param.delta_index = 0.0 +/- 0.1
param.delta_index.fixed = False
param.delta_index.range = nominal + [-5, +5] * sigma

[osc.prob3]
calc_mode = true_allsky_fine
apply_mode = events
param.earth_model = osc/PREM_12layer.dat
param.YeI = ${earth:YeI}
param.YeM = ${earth:YeM}
param.YeO = ${earth:YeO}
param.detector_depth = ${earth:detector_depth}
param.prop_height = ${earth:prop_height}
param.theta12 = ${osc:theta12}
param.theta12.fixed = True
param.nh.theta13 = ${osc:theta13_nh}
param.nh.theta13.fixed = False
param.nh.theta13.range = ${osc:theta13_nh.range}
param.ih.theta13 = ${osc:theta13_ih}
param.ih.theta13.fixed = False
param.ih.theta13.range = ${osc:theta13_ih.range}
param.nh.theta23 = ${osc:theta23_nh}
param.nh.theta23.fixed = False
param.nh.theta23.range = ${osc:theta23_nh.range}
param.nh.theta23.prior = uniform
param.ih.theta23 = ${osc:theta23_ih}
param.ih.theta23.fixed = False
param.ih.theta23.range = ${osc:theta23_ih.range}
param.ih.theta23.prior = uniform
param.nh.deltacp = 0.0 * units.degree
param.nh.deltacp.fixed = True
param.nh.deltacp.range = ${osc:deltacp_nh.range}
param.nh.deltacp.prior = uniform
param.ih.deltacp = 0.0 * units.degree
param.ih.deltacp.fixed = True
param.deltam21 = ${osc:deltam21}
param.deltam21.fixed = True
param.nh.deltam31 = ${osc:deltam31_nh}
param.nh.deltam31.fixed = False
param.nh.deltam31.prior = uniform
param.nh.deltam31.range = [0.001, +0.007] * units.eV**2
param.ih.deltam31 = ${osc:deltam31_ih}
param.ih.deltam31.fixed = False
param.ih.deltam31.prior = uniform
param.ih.deltam31.range = [-0.007, -0.001] * units.eV**2

[aeff.aeff]
calc_mode = events
apply_mode = events
param.livetime = 11.7 * units.common_year
param.aeff_scale = 1.0
param.aeff_scale.fixed = False
param.aeff_scale.prior = uniform
param.aeff_scale.range = [0.,3.] * units.dimensionless
param.nutau_cc_norm = 1.0
param.nutau_cc_norm.fixed = True
param.nutau_cc_norm.range = [0.2, 2.0] * units.dimensionless
param.nutau_cc_norm.prior = uniform
param.nutau_norm = 1.0
param.nutau_norm.fixed = False
param.nutau_norm.range = [-1.0, 8.5] * units.dimensionless
param.nutau_norm.prior = uniform
param.nu_nc_norm = 1.0 +/- 0.2
param.nu_nc_norm.fixed = False
param.nu_nc_norm.range = nominal + [-.5,+.5]

[utils.hist]
calc_mode = events
apply_mode = graphnet_dynamic_binning
error_method = sumw2"""%(config_dict['binning_cfg'],
                            config_dict['post_fix'],
                            config_dict['pipeline'])
    if config_dict['post_fix'] != None:
        filename = 'pipeline_config_%s.cfg'%config_dict['post_fix']
        with open(outdir + '/' + filename, 'w+') as f:
            f.write(template)
    else:
        filename = 'pipeline_config.cfg'
        with open(outdir + '/' + filename, 'w+') as f:
            f.write(template)
    return outdir + '/' + filename

def make_configs(outdir, run_name, is_retro, pipeline_path, post_fix = '_pred', config_dict = None):
    os.makedirs(outdir + '/' + run_name, exist_ok=True)
    if config_dict == None:
        # Run on standard settings
        config_dict = {'reco_energy' : {'num_bins': 8},
                       'reco_coszen' : {'num_bins': 8},
                       'pid'         : {'bin_edges': [0,0.5,1]},
                       'true_energy' : {'num_bins': 200},
                       'true_coszen' : {'num_bins': 200}}
    
    config_dict['pipeline'] = pipeline_path
    if is_retro:
        config_dict['post_fix'] = '_retro'
    else:
        config_dict['post_fix'] = post_fix
    binning_cfg_path = make_binning_cfg(config_dict, outdir + '/' + run_name)
    config_dict['binning_cfg'] = binning_cfg_path
    pipeline_cfg_path = make_pipeline_cfg(config_dict, outdir + '/' + run_name)
    return pipeline_cfg_path

def parallel_fit_2D_contour(settings):
    """fitting routine for 2D contours. Length of settings determines the amount of jobs this worker gets.

        Results are saved to temporary .csv-files that are later merged.

    Args:
        settings (list): A list of fitting settings.
    """    
    results = []
    for i in range(len(settings)):
        cfg_path, model_name, outdir, theta23_value, deltam31_value, id, run_name, fix_all, minimizer_cfg = settings[i]
        minimizer_cfg = pisa.utils.fileio.from_file(minimizer_cfg)
        model = DistributionMaker([cfg_path])
        data = model.get_outputs(return_sum=True)
        ana = Analysis()
        if fix_all == 'True':
            # Only free parameters will be [parameter, aeff_scale] - corresponding to a statistical fit
            free_params = model.params.free.names
            for free_param in free_params:
                if free_param != 'aeff_scale':
                    if free_param == 'theta23':
                        model.params.theta23.is_fixed = True
                        model.params.theta23.nominal_value = float(theta23_value) * ureg.degree
                    elif free_param == 'deltam31':
                        model.params.deltam31.is_fixed = True
                        model.params.deltam31.nominal_value = float(deltam31_value) * ureg.electron_volt ** 2
                    else:
                        model.params[free_param].is_fixed = True
        else:
            # Only fixed parameters will be [parameter]
            model.params.theta23.is_fixed = True
            model.params.deltam31.is_fixed = True
            model.params.theta23.nominal_value = float(theta23_value) * ureg.degree
            model.params.deltam31.nominal_value = float(deltam31_value) * ureg.electron_volt ** 2
        model.reset_all()
        result = ana.fit_hypo(
            data,
            model,
            metric='mod_chi2',
            minimizer_settings=minimizer_cfg,
            fit_octants_separately=True,
            )
        results.append([theta23_value, deltam31_value, result[0]['params'].theta23.value, result[0]['params'].deltam31.value, result[0]['metric_val'], model_name, id, result[0]['minimizer_metadata']['success']])
    os.makedirs(outdir + '/' + run_name + '/tmp', exist_ok = True)
    results = pd.DataFrame(data = results, columns = ['theta23_fixed', 'dm31_fixed', 'theta23_best_fit', 'dm31_best_fit', 'mod_chi2', 'model', 'id', 'converged'])
    results.to_csv(outdir + '/' + run_name + '/tmp' + '/tmp_%s.csv'%id)
    return

def parallel_fit_1D_contour(settings):
    """fitting routine for 1D contours. Length of settings determines the amount of jobs this worker gets.

        Results are saved to temporary .csv-files that are later merged.

    Args:
        settings (list): A list of fitting settings.
    """
    results = []
    for i in range(len(settings)):
        cfg_path, model_name, outdir, theta23_value, deltam31_value, id, run_name, parameter, fix_all, minimizer_cfg = settings[i]
        minimizer_cfg = pisa.utils.fileio.from_file(minimizer_cfg)
        ana = Analysis()
        model = DistributionMaker([cfg_path])
        data = model.get_outputs(return_sum=True)
        if  fix_all=='True':
            # Only free parameters will be [parameter, aeff_scale] - corresponding to a statistical fit
            free_params = model.params.free.names
            for free_param in free_params:
                if free_param not in  ['aeff_scale', 'theta23', 'deltam31']:
                    model.params[free_param].is_fixed = True
            if parameter == 'theta23':
                model.params.theta23.is_fixed = True
                model.params.theta23.nominal_value = float(theta23_value) * ureg.degree
            elif parameter == 'deltam31':
                model.params.deltam31.is_fixed = True
                model.params.deltam31.nominal_value = float(deltam31_value) * ureg.electron_volt ** 2
        else:
            # Only fixed parameters will be [parameter]
            if parameter == 'theta23':
                model.params.theta23.is_fixed = True
                model.params.theta23.nominal_value = float(theta23_value) * ureg.degree
            elif parameter == 'deltam31':
                model.params.deltam31.is_fixed = True
                model.params.deltam31.nominal_value = float(deltam31_value) * ureg.electron_volt ** 2
            else:
                print('parameter not supported: %s'%parameter)
        model.reset_all()
        result = ana.fit_hypo(
            data,
            model,
            metric='mod_chi2',
            minimizer_settings=minimizer_cfg,
            fit_octants_separately=True,
            )
        results.append([theta23_value, deltam31_value, result[0]['params'].theta23.value, result[0]['params'].deltam31.value, result[0]['metric_val'], model_name, id,result[0]['minimizer_metadata']['success']])

    os.makedirs(outdir + '/' + run_name + '/tmp', exist_ok = True)
    results = pd.DataFrame(data = results, columns = ['theta23_fixed', 'dm31_fixed', 'theta23_best_fit', 'dm31_best_fit', 'mod_chi2', 'model','id', 'converged'])
    results.to_csv(outdir + '/' + run_name + '/tmp' + '/tmp_%s.csv'%id)
    return

def merge_temporary_files(outdir, run_name):
    files = os.listdir(outdir + '/' + run_name + '/tmp')
    is_first = True
    for file in files:
        if is_first:
            df = pd.read_csv(outdir + '/' + run_name +'/tmp/' +  file)
            is_first = False
        else:
            df = df.append(pd.read_csv(outdir + '/' + run_name + '/tmp/' +  file), ignore_index = True)
    df = df.reset_index(drop = True)
    return df

def calculate_2D_contours(outdir, run_name, pipeline_path, post_fix = '_pred',model_name = 'gnn', include_retro = True, config_dict = None, grid_size = 30, n_workers = 10, statistical_fit = False):
    """Calculate 2D contours for mixing angle theta_23 and mass difference dm31. Results are saved to outdir/merged_results.csv
    
    Args:
        outdir (str): path to directory where contour data is stored.
        run_name (str): name of the folder that will be created in outdir.
        pipeline_path (str): path to InSQLite pipeline database.
        model_name (str): name of the GNN. Defaults to 'gnn'.
        include_retro (bool): If True, contours for retro will also be included. Defaults to True.
        config_dict (dict): dictionary with pisa settings. Allows the user to overwrite binning decisions and minimizer choice. If None, the fitting is run using standard configuration. Defaults to None.
        grid_size (int, optional): Number of points fitted in each oscillation variable. grid_size = 10 means 10*10 points fitted. Defaults to 30.
        n_workers (int, optional): Number of parallel fitting routines. Cannot be larger than the number of fitting points. Defaults to 10.
        statistical_fit (bool, optional): Will fit only aeff_scale if True. Defaults to False.
    """
    if 'minimizer_cfg' in config_dict.keys():
        minimizer_cfg = config_dict['minimizer_cfg']
    else:
        root = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        minimizer_cfg = root + '/resources/settings/minimizer/graphnet_standard.json'
    cfgs = {}
    cfgs[model_name] = make_configs(outdir = outdir, post_fix = post_fix, run_name = run_name,is_retro = False, pipeline_path = pipeline_path, config_dict = config_dict)
    if include_retro:
        cfgs['retro'] = make_configs(outdir = outdir, post_fix = post_fix, run_name = run_name,is_retro = True, pipeline_path = pipeline_path, config_dict = config_dict)
    statistical_fit = str(statistical_fit) # When sent to workers, booleans can be messed up. Converting to strings are more robust.
    theta23_range = np.linspace(36, 54, grid_size)
    dm31_range = np.linspace(2.3,2.7, grid_size)*1e-3
    settings = []
    count = 0
    for model_name in cfgs.keys():
        for i in range(0,grid_size):
            for k in range(0,grid_size):
                settings.append([cfgs[model_name], model_name, outdir, theta23_range[i],dm31_range[k], count, run_name, statistical_fit, minimizer_cfg])
                count +=1
    random.shuffle(settings)
    chunked_settings = np.array_split(settings, n_workers)
    #parallel_fit_2D_contour(chunked_settings[0]) # for debugging
    p = multiprocessing.Pool(processes = len(chunked_settings))
    _ = p.map_async(parallel_fit_2D_contour,chunked_settings)
    p.close()
    p.join()
    df = merge_temporary_files(outdir, run_name)
    df.to_csv(outdir + '/' + run_name + '/merged_results.csv')
    return

def calculate_1D_contours(outdir, run_name, pipeline_path,  post_fix = '_pred', model_name = 'gnn', include_retro = True, config_dict = None, grid_size = 30, n_workers = 10, statistical_fit = False):
    """Calculate 1D contours for mixing angle theta_23 and mass difference dm31. Results are saved to outdir/merged_results.csv

    Args:
        outdir (str): path to directory where contour data is stored.
        run_name (str): name of the folder that will be created in outdir.
        pipeline_path (str): path to InSQLite pipeline database.
        model_name (str): name of the GNN. Defaults to 'gnn'.
        include_retro (bool): If True, contours for retro will also be included. Defaults to True.
        config_dict (dict): dictionary with pisa settings. Allows the user to overwrite binning decisions and minimizer choice. If None, the fitting is run using standard configuration. Defaults to None.
        grid_size (int, optional): Number of points fitted in each oscillation variable. grid_size = 10 means 10*10 points fitted. Defaults to 30.
        n_workers (int, optional): Number of parallel fitting routines. Cannot be larger than the number of fitting points. Defaults to 10.
        statistical_fit (bool, optional): Will fit only aeff_scale if True. Defaults to False.
    """
    if config_dict != None and 'minimizer_cfg' in config_dict.keys():
        minimizer_cfg = config_dict['minimizer_cfg']
    else:
        root = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        minimizer_cfg = root + '/resources/settings/minimizer/graphnet_standard.json'
    cfgs = {}
    cfgs[model_name] = make_configs(outdir = outdir, post_fix = post_fix, run_name = run_name, is_retro = False, pipeline_path = pipeline_path, config_dict = config_dict)
    if include_retro:
        cfgs['retro'] = make_configs(outdir = outdir, post_fix = post_fix, run_name = run_name, is_retro = True, pipeline_path = pipeline_path, config_dict = config_dict)
    statistical_fit = str(statistical_fit) # When sent to workers, booleans can be messed up. Converting to strings are more robust.
    theta23_range = np.linspace(36, 54, grid_size)
    dm31_range = np.linspace(2.3,2.7, grid_size)*1e-3
    settings = []
    count = 0
    for model_name in cfgs.keys():
        for i in range(0,grid_size):
            settings.append([cfgs[model_name], model_name, outdir, theta23_range[i],-1, count, run_name, 'theta23',statistical_fit, minimizer_cfg])
            count +=1
        for i in range(0,grid_size):
            settings.append([cfgs[model_name], model_name, outdir, -1,dm31_range[i], count, run_name, 'deltam31',statistical_fit, minimizer_cfg])
            count +=1
    random.shuffle(settings)
    chunked_settings = np.array_split(settings, n_workers)
    #parallel_fit_1D_contour(chunked_settings[0]) # for debugging
    p = multiprocessing.Pool(processes = len(chunked_settings))
    _ = p.map_async(parallel_fit_1D_contour,chunked_settings)
    p.close()
    p.join()
    df = merge_temporary_files(outdir, run_name)
    df.to_csv(outdir + '/' + run_name + '/merged_results.csv')
    return

