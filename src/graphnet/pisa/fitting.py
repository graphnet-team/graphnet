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
import matplotlib as mpl
import matplotlib.pyplot as plt
import configparser
import io
from configupdater import ConfigUpdater
from contextlib import contextmanager

mpl.use("pdf")
plt.rc("font", family="serif")


@contextmanager
def config_updater(
    config_path: str,
    new_config_path: str = None,
    dummy_section: str = "temp",
) -> ConfigUpdater:
    """Updates config files and saves them to file.

    Args:
        config_path (str): Path to original config file.
        new_config_path (str, optional): Path to save updated config file.
            Defaults to None.
        dummy_section (str, optional): Dummy section name to use for config
            files without section headers. Defaults to "temp".

    Yields:
        ConfigUpdater: Instance for programatically updating config file.
    """
    # Modify original config file is no new config path is provided.
    if new_config_path is None:
        new_config_path = config_path

    # Load config file
    updater = ConfigUpdater()
    has_dummy_section = False
    try:
        updater.read(config_path)

    # If it is missing section headers (e.g., binning.cfg), add a dummy section
    # header before reading file contents.
    except configparser.MissingSectionHeaderError:
        with open(config_path, "r") as configfile:
            updater.read_string(f"[{dummy_section}]\n" + configfile.read())
        has_dummy_section = True

    # Expose updater instance in contest (i.e.,
    # `with config_updater(...) as updater:``).
    try:
        yield updater

    # Write new config to file
    finally:
        with open(new_config_path, "w") as configfile:
            if has_dummy_section:
                # Removing dummy section header if necessary
                with io.StringIO() as buffer:
                    updater.write(buffer)
                    buffer.seek(0)
                    lines = buffer.readlines()[1:]
                configfile.writelines(lines)
            else:
                updater.write(configfile)


class ContourFitter:
    def __init__(
        self,
        outdir,
        pipeline_path,
        post_fix="_pred",
        model_name="gnn",
        include_retro=True,
        statistical_fit=False,
    ):
        self._outdir = outdir
        self._pipeline_path = pipeline_path
        self._post_fix = post_fix
        self._model_name = model_name
        self._include_retro = include_retro
        self._statistical_fit = str(statistical_fit)
        self._allowed_contour_types = ["1d", "2d"]

    def fit_1d_contour(
        self,
        run_name,
        config_dict,
        grid_size=30,
        n_workers=1,
        theta23_minmax=[36, 54],
        dm31_minmax=[2.3, 2.7],
    ):
        self._fit_contours(
            config_dict=config_dict,
            run_name=run_name,
            grid_size=grid_size,
            n_workers=n_workers,
            theta23_minmax=theta23_minmax,
            dm31_minmax=dm31_minmax,
            contour_type="1d",
        )
        return

    def fit_2d_contour(
        self,
        run_name,
        config_dict,
        grid_size=30,
        n_workers=1,
        theta23_minmax=[36, 54],
        dm31_minmax=[2.3, 2.7],
    ):
        self._fit_contours(
            config_dict=config_dict,
            run_name=run_name,
            grid_size=grid_size,
            n_workers=n_workers,
            theta23_minmax=theta23_minmax,
            dm31_minmax=dm31_minmax,
            contour_type="2d",
        )
        return

    def _check_inputs(
        self, contour_type, dm31_minmax, theta23_minmax, n_workers
    ):
        if contour_type.lower() not in self._allowed_contour_types:
            print(
                "%s not recognized as valid contour type. Only %s is recognized"
                % (contour_type, self._allowed_contour_types)
            )
            return False
        if (
            (len(theta23_minmax) != 2)
            or (len(dm31_minmax) != 2)
            or (dm31_minmax[0] > dm31_minmax[1])
            or (theta23_minmax[0] > theta23_minmax[1])
        ):
            print(
                "theta23 or dm31 min max values are not understood. Please provide a list on the form [min, max] for both variables"
            )
            return False
        if n_workers < 1:
            print("found n_workers < 1. n_workers must be positive integers.")
            return False
        return True

    def _fit_contours(
        self,
        run_name,
        config_dict,
        grid_size,
        n_workers,
        contour_type,
        theta23_minmax,
        dm31_minmax,
    ):
        inputs_ok = self._check_inputs(
            contour_type=contour_type,
            dm31_minmax=dm31_minmax,
            theta23_minmax=theta23_minmax,
            n_workers=n_workers,
        )
        if inputs_ok:
            minimizer_cfg = self._get_minimizer_path(config_dict)
            cfgs = self._setup_config_files(run_name, config_dict)
            theta23_range = np.linspace(
                theta23_minmax[0], theta23_minmax[1], grid_size
            )
            dm31_range = (
                np.linspace(dm31_minmax[0], dm31_minmax[1], grid_size) * 1e-3
            )
            if contour_type.lower() == "1d":
                settings = self._make_1d_settings(
                    cfgs=cfgs,
                    grid_size=grid_size,
                    run_name=run_name,
                    minimizer_cfg=minimizer_cfg,
                    theta23_range=theta23_range,
                    dm31_range=dm31_range,
                    n_workers=n_workers,
                )
                p = multiprocessing.Pool(processes=len(settings))
                _ = p.map_async(self._parallel_fit_1d_contour, settings)
                p.close()
                p.join()
                # self._parallel_fit_1d_contour(settings[0])
            elif contour_type.lower() == "2d":
                settings = self._make_2d_settings(
                    cfgs=cfgs,
                    grid_size=grid_size,
                    run_name=run_name,
                    minimizer_cfg=minimizer_cfg,
                    theta23_range=theta23_range,
                    dm31_range=dm31_range,
                    n_workers=n_workers,
                )
                p = multiprocessing.Pool(processes=len(settings))
                _ = p.map_async(self._parallel_fit_2d_contour, settings)
                p.close()
                p.join()
                # self._parallel_fit_2d_contour(settings[0])
            df = self._merge_temporary_files(run_name)
            df.to_csv(self._outdir + "/" + run_name + "/merged_results.csv")
        else:
            return

    def _merge_temporary_files(self, run_name):
        files = os.listdir(self._outdir + "/" + run_name + "/tmp")
        is_first = True
        for file in files:
            if is_first:
                df = pd.read_csv(
                    self._outdir + "/" + run_name + "/tmp/" + file
                )
                is_first = False
            else:
                df = df.append(
                    pd.read_csv(
                        self._outdir + "/" + run_name + "/tmp/" + file
                    ),
                    ignore_index=True,
                )
        df = df.reset_index(drop=True)
        return df

    def _parallel_fit_2d_contour(self, settings):
        """fitting routine for 2D contours. Length of settings determines the amount of jobs this worker gets.

            Results are saved to temporary .csv-files that are later merged.

        Args:
            settings (list): A list of fitting settings.
        """
        results = []
        for i in range(len(settings)):
            (
                cfg_path,
                model_name,
                outdir,
                theta23_value,
                deltam31_value,
                id,
                run_name,
                fix_all,
                minimizer_cfg,
            ) = settings[i]
            minimizer_cfg = pisa.utils.fileio.from_file(minimizer_cfg)
            model = DistributionMaker([cfg_path])
            data = model.get_outputs(return_sum=True)
            ana = Analysis()
            if fix_all == "True":
                # Only free parameters will be [parameter, aeff_scale] - corresponding to a statistical fit
                free_params = model.params.free.names
                for free_param in free_params:
                    if free_param != "aeff_scale":
                        if free_param == "theta23":
                            model.params.theta23.is_fixed = True
                            model.params.theta23.nominal_value = (
                                float(theta23_value) * ureg.degree
                            )
                        elif free_param == "deltam31":
                            model.params.deltam31.is_fixed = True
                            model.params.deltam31.nominal_value = (
                                float(deltam31_value) * ureg.electron_volt**2
                            )
                        else:
                            model.params[free_param].is_fixed = True
            else:
                # Only fixed parameters will be [parameter]
                model.params.theta23.is_fixed = True
                model.params.deltam31.is_fixed = True
                model.params.theta23.nominal_value = (
                    float(theta23_value) * ureg.degree
                )
                model.params.deltam31.nominal_value = (
                    float(deltam31_value) * ureg.electron_volt**2
                )
            model.reset_all()
            result = ana.fit_hypo(
                data,
                model,
                metric="mod_chi2",
                minimizer_settings=minimizer_cfg,
                fit_octants_separately=True,
            )
            results.append(
                [
                    theta23_value,
                    deltam31_value,
                    result[0]["params"].theta23.value,
                    result[0]["params"].deltam31.value,
                    result[0]["metric_val"],
                    model_name,
                    id,
                    result[0]["minimizer_metadata"]["success"],
                ]
            )
        self._save_temporary_results(
            outdir=outdir, run_name=run_name, results=results, id=id
        )
        return

    def _parallel_fit_1d_contour(self, settings):
        """fitting routine for 1D contours. Length of settings determines the amount of jobs this worker gets.

            Results are saved to temporary .csv-files that are later merged.

        Args:
            settings (list): A list of fitting settings.
        """
        results = []
        for i in range(len(settings)):
            (
                cfg_path,
                model_name,
                outdir,
                theta23_value,
                deltam31_value,
                id,
                run_name,
                parameter,
                fix_all,
                minimizer_cfg,
            ) = settings[i]
            minimizer_cfg = pisa.utils.fileio.from_file(minimizer_cfg)
            ana = Analysis()
            model = DistributionMaker([cfg_path])
            data = model.get_outputs(return_sum=True)
            if fix_all == "True":
                # Only free parameters will be [parameter, aeff_scale] - corresponding to a statistical fit
                free_params = model.params.free.names
                for free_param in free_params:
                    if free_param not in ["aeff_scale", "theta23", "deltam31"]:
                        model.params[free_param].is_fixed = True
                if parameter == "theta23":
                    model.params.theta23.is_fixed = True
                    model.params.theta23.nominal_value = (
                        float(theta23_value) * ureg.degree
                    )
                elif parameter == "deltam31":
                    model.params.deltam31.is_fixed = True
                    model.params.deltam31.nominal_value = (
                        float(deltam31_value) * ureg.electron_volt**2
                    )
            else:
                # Only fixed parameters will be [parameter]
                if parameter == "theta23":
                    model.params.theta23.is_fixed = True
                    model.params.theta23.nominal_value = (
                        float(theta23_value) * ureg.degree
                    )
                elif parameter == "deltam31":
                    model.params.deltam31.is_fixed = True
                    model.params.deltam31.nominal_value = (
                        float(deltam31_value) * ureg.electron_volt**2
                    )
                else:
                    print("parameter not supported: %s" % parameter)
            model.reset_all()
            result = ana.fit_hypo(
                data,
                model,
                metric="mod_chi2",
                minimizer_settings=minimizer_cfg,
                fit_octants_separately=True,
            )
            results.append(
                [
                    theta23_value,
                    deltam31_value,
                    result[0]["params"].theta23.value,
                    result[0]["params"].deltam31.value,
                    result[0]["metric_val"],
                    model_name,
                    id,
                    result[0]["minimizer_metadata"]["success"],
                ]
            )
        self._save_temporary_results(
            outdir=outdir, run_name=run_name, results=results, id=id
        )
        return

    def _save_temporary_results(self, outdir, run_name, results, id):
        os.makedirs(outdir + "/" + run_name + "/tmp", exist_ok=True)
        results = pd.DataFrame(
            data=results,
            columns=[
                "theta23_fixed",
                "dm31_fixed",
                "theta23_best_fit",
                "dm31_best_fit",
                "mod_chi2",
                "model",
                "id",
                "converged",
            ],
        )
        results.to_csv(outdir + "/" + run_name + "/tmp" + "/tmp_%s.csv" % id)
        return

    def _make_2d_settings(
        self,
        cfgs,
        grid_size,
        run_name,
        minimizer_cfg,
        theta23_range,
        dm31_range,
        n_workers,
    ):
        settings = []
        count = 0
        for model_name in cfgs.keys():
            for i in range(0, grid_size):
                for k in range(0, grid_size):
                    settings.append(
                        [
                            cfgs[model_name],
                            model_name,
                            self._outdir,
                            theta23_range[i],
                            dm31_range[k],
                            count,
                            run_name,
                            self._statistical_fit,
                            minimizer_cfg,
                        ]
                    )
                    count += 1
        random.shuffle(settings)
        return np.array_split(settings, n_workers)

    def _make_1d_settings(
        self,
        cfgs,
        grid_size,
        run_name,
        minimizer_cfg,
        theta23_range,
        dm31_range,
        n_workers,
    ):
        settings = []
        count = 0
        for model_name in cfgs.keys():
            for i in range(0, grid_size):
                settings.append(
                    [
                        cfgs[model_name],
                        model_name,
                        self._outdir,
                        theta23_range[i],
                        -1,
                        count,
                        run_name,
                        "theta23",
                        self._statistical_fit,
                        minimizer_cfg,
                    ]
                )
                count += 1
            for i in range(0, grid_size):
                settings.append(
                    [
                        cfgs[model_name],
                        model_name,
                        self._outdir,
                        -1,
                        dm31_range[i],
                        count,
                        run_name,
                        "deltam31",
                        self._statistical_fit,
                        minimizer_cfg,
                    ]
                )
                count += 1
        random.shuffle(settings)
        return np.array_split(settings, n_workers)

    def _setup_config_files(self, run_name, config_dict):
        cfgs = {}
        cfgs[self._model_name] = self._make_configs(
            outdir=self._outdir,
            post_fix=self._post_fix,
            run_name=run_name,
            is_retro=False,
            pipeline_path=self._pipeline_path,
            config_dict=config_dict,
        )
        if self._include_retro:
            cfgs["retro"] = self._make_configs(
                outdir=self._outdir,
                post_fix=self._post_fix,
                run_name=run_name,
                is_retro=True,
                pipeline_path=self._pipeline_path,
                config_dict=config_dict,
            )
        return cfgs

    def _get_minimizer_path(self, config_dict):
        if config_dict is not None and "minimizer_cfg" in config_dict.keys():
            minimizer_cfg = config_dict["minimizer_cfg"]
        else:
            root = os.path.realpath(
                os.path.join(os.getcwd(), os.path.dirname(__file__))
            )
            minimizer_cfg = (
                root + "/resources/minimizer/graphnet_standard.json"
            )
        return minimizer_cfg

    def _create_configs(self, config_dict, path):
        # Update binning config
        root = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        if config_dict["post_fix"] is not None:
            config_name = "config%s" % config_dict["post_fix"]
        else:
            config_name = "config"

        with config_updater(
            root
            + "/resources/configuration_templates/binning_config_template.cfg",
            "%s/binning_%s.cfg" % (path, config_name),
            dummy_section="binning",
        ) as updater:
            updater["binning"][
                "graphnet_dynamic_binning.reco_energy"
            ].value = (
                "{'num_bins':%s, 'is_log':True, 'domain':[0.5,55] * units.GeV, 'tex': r'E_{\\rm reco}'}"
                % config_dict["reco_energy"]["num_bins"]
            )  # noqa: W605
            updater["binning"][
                "graphnet_dynamic_binning.reco_coszen"
            ].value = (
                "{'num_bins':%s, 'is_lin':True, 'domain':[-1,1], 'tex':r'\\cos{\\theta}_{\\rm reco}'}"
                % config_dict["reco_coszen"]["num_bins"]
            )  # noqa: W605
            updater["binning"]["graphnet_dynamic_binning.pid"].value = (
                "{'bin_edges': %s, 'tex':r'{\\rm PID}'}"
                % config_dict["pid"]["bin_edges"]
            )  # noqa: W605
            updater["binning"]["true_allsky_fine.true_energy"].value = (
                "{'num_bins':%s, 'is_log':True, 'domain':[1,1000] * units.GeV, 'tex': r'E_{\\rm true}'}"
                % config_dict["true_energy"]["num_bins"]
            )  # noqa: W605
            updater["binning"]["true_allsky_fine.true_coszen"].value = (
                "{'num_bins':%s, 'is_lin':True, 'domain':[-1,1], 'tex':r'\\cos\,\\theta_{Z,{\\rm true}}'}"  # noqa: W605
                % config_dict["true_coszen"]["num_bins"]
            )  # noqa: W605

        # Update pipeline config
        with config_updater(
            root
            + "/resources/configuration_templates/pipeline_config_template.cfg",
            "%s/pipeline_%s.cfg" % (path, config_name),
        ) as updater:
            updater["pipeline"].add_before.comment(
                "#include %s/binning_%s.cfg as binning" % (path, config_name)
            )
            updater["data.sqlite_loader"]["post_fix"].value = config_dict[
                "post_fix"
            ]
            updater["data.sqlite_loader"]["database"].value = config_dict[
                "pipeline"
            ]
            if "livetime" in config_dict.keys():
                updater["aeff.aeff"]["param.livetime"].value = (
                    "%s * units.common_year" % config_dict["livetime"]
                )
        return "%s/pipeline_%s.cfg" % (path, config_name)

    def _make_configs(
        self,
        outdir,
        run_name,
        is_retro,
        pipeline_path,
        post_fix="_pred",
        config_dict=None,
    ):
        os.makedirs(outdir + "/" + run_name, exist_ok=True)
        if config_dict is None:
            # Run on standard settings
            config_dict = {
                "reco_energy": {"num_bins": 8},
                "reco_coszen": {"num_bins": 8},
                "pid": {"bin_edges": [0, 0.5, 1]},
                "true_energy": {"num_bins": 200},
                "true_coszen": {"num_bins": 200},
                "livetime": 10,
            }

        config_dict["pipeline"] = pipeline_path
        if is_retro:
            config_dict["post_fix"] = "_retro"
        else:
            config_dict["post_fix"] = post_fix
        pipeline_cfg_path = self._create_configs(
            config_dict, outdir + "/" + run_name
        )
        return pipeline_cfg_path
