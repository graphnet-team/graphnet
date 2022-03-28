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

def parallel_fit_2D_contour(settings):
    """fitting routine for 2D contours. Length of settings determines the amount of jobs this worker gets.

        Results are saved to temporary .csv-files that are later merged.

    Args:
        settings (list): A list of fitting settings.
    """    
    results = []
    for i in range(len(settings)):
        print(i)
        cfg_path, model_name, outdir, theta23_value, deltam31_value, id, run_name, fix_all, minimizer_cfg = settings[i]
        model = DistributionMaker([cfg_path])
        data = model.get_outputs(return_sum=True)
        ana = Analysis()
        if fix_all == True:
            # Only free parameters will be [parameter, aeff_scale] - corresponding to a statistical fit
            free_params = model.params.free.names
            for free_param in free_params:
                if free_param != 'aeff_scale':
                    print(free_param)
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
                    print(free_param)
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
        os.makedirs(outdir + '/' + run_name + '/fit_objects', exist_ok = True)

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

def calculate_2D_contours(outdir, cfgs, run_name, minimizer_cfg, grid_size = 30, n_workers = 10, statistical_fit = False):
    """Calculate 2D contours for mixing angle theta_23 and mass difference dm31. Results are saved to outdir/merged_results.csv
    
    Args:
        outdir (str): path to directory where contour data is stored
        cfgs (dict): dictionary containing model name and corresponding config file. e.g. {'retro': '/retro_cfg.cfg'}
        run_name (str): name of the folder that will be created in outdir
        minimizer_cfg (str): path to minimizer cfg
        grid_size (int, optional): Number of points fitted in each oscillation variable. grid_size = 10 means 10*10 points fitted. Defaults to 30.
        n_workers (int, optional): Number of parallel fitting routines. Cannot be larger than the number of fitting points. Defaults to 10.
        statistical_fit (bool, optional): Will fit only aeff_scale if True. Defaults to False.
    """
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
    #parallel_fit_contour(chunked_settings[0]) # for debugging
    p = multiprocessing.Pool(processes = len(chunked_settings))
    _ = p.map_async(parallel_fit_2D_contour,chunked_settings)
    p.close()
    p.join()
    df = merge_temporary_files(outdir, run_name)
    df.to_csv(outdir + '/' + run_name + '/merged_results.csv')
    return

def calculate_1D_contours(outdir, cfgs, run_name, minimizer_cfg,grid_size = 30, n_workers = 10, statistical_fit = False):
    """Calculate 1D contours for mixing angle theta_23 and mass difference dm31. Results are saved to outdir/merged_results.csv

    Args:
        outdir (str): path to directory where contour data is stored
        cfgs (dict): dictionary containing model name and corresponding config file. e.g. {'retro': '/retro_cfg.cfg'}
        run_name (str): name of the folder that will be created in outdir
        minimizer_cfg (str): path to minimizer cfg
        grid_size (int, optional): Number of points fitted in the 1D contours. Defaults to 30.
        n_workers (int, optional): Number of parallel fitting routines. Cannot be larger than the number of fitting points. Defaults to 10.
        statistical_fit (bool, optional): Will fit only aeff_scale if True. Defaults to False.
    """
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
    #parallel_fit_single_parameter(chunked_settings[0]) # for debugging
    p = multiprocessing.Pool(processes = len(chunked_settings))
    _ = p.map_async(parallel_fit_1D_contour,chunked_settings)
    p.close()
    p.join()
    df = merge_temporary_files(outdir, run_name)
    df.to_csv(outdir + '/' + run_name + '/merged_results.csv')
    return

