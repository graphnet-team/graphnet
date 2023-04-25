"""Functions and classes for fitting contours using PISA."""

import configparser
from contextlib import contextmanager
import io
import multiprocessing
import os
import random
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from configupdater import ConfigUpdater
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graphnet.utilities.imports import has_pisa_package

if has_pisa_package() or TYPE_CHECKING:
    import pisa  # pyright: reportMissingImports=false
    from pisa.core.distribution_maker import DistributionMaker
    from pisa.core.pipeline import Pipeline
    from pisa.analysis.analysis import Analysis
    from pisa import ureg

from graphnet.data.sqlite import create_table_and_save_to_sql

mpl.use("pdf")
plt.rc("font", family="serif")


@contextmanager
def config_updater(
    config_path: str,
    new_config_path: Optional[str] = None,
    dummy_section: str = "temp",
) -> ConfigUpdater:
    """Update config files and saves them to file.

    Args:
        config_path: Path to original config file.
        new_config_path: Path to save updated config file.
        dummy_section: Dummy section name to use for config files without
            section headers.

    Yields:
        ConfigUpdater instance for programatically updating config file.
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


class WeightFitter:
    """Class for fitting weights using PISA."""

    def __init__(
        self,
        database_path: str,
        truth_table: str = "truth",
        index_column: str = "event_no",
        statistical_fit: bool = False,
    ) -> None:
        """Construct `WeightFitter`."""
        self._database_path = database_path
        self._truth_table = truth_table
        self._index_column = index_column
        self._statistical_fit = statistical_fit

    def fit_weights(
        self,
        config_outdir: str,
        weight_name: str = "",
        pisa_config_dict: Optional[Dict] = None,
        add_to_database: bool = False,
    ) -> pd.DataFrame:
        """Fit flux weights to each neutrino event in `self._database_path`.

        If `statistical_fit=True`, only statistical effects are accounted for.
        If `True`, certain systematic effects are included, but not
        hypersurfaces.

        Args:
            config_outdir: The output directory in which to store the
                configuration.
            weight_name: The name of the weight. If `add_to_database=True`,
                this will be the name of the table.
            pisa_config_dict: The dictionary of PISA configurations. Can be
                used to change assumptions regarding the fit.
            add_to_database: If `True`, a table will be added to the database
                called `weight_name` with two columns:
                `[index_column, weight_name]`

        Returns:
            A dataframe with columns `[index_column, weight_name]`.
        """
        # If its a standard weight
        if pisa_config_dict is None:
            if not weight_name:
                print(weight_name)
                weight_name = "pisa_weight_graphnet_standard"

        # If it is a custom weight without name
        elif pisa_config_dict is not None:
            if not weight_name:
                weight_name = "pisa_custom_weight"

        pisa_config_path = self._make_config(
            config_outdir, weight_name, pisa_config_dict
        )

        model = Pipeline(pisa_config_path)

        if self._statistical_fit == "True":
            # Only free parameters will be [aeff_scale] - corresponding to a statistical fit
            free_params = model.params.free.names
            for free_param in free_params:
                if free_param not in ["aeff_scale"]:
                    model.params[free_param].is_fixed = True

        # for stage in range(len(model.stages)):
        model.stages[-1].apply_mode = "events"
        model.stages[-1].calc_mode = "events"
        model.run()

        all_data = []
        for container in model.data:
            data = pd.DataFrame(container["event_no"], columns=["event_no"])
            data[weight_name] = container["weights"]
            all_data.append(data)
        results = pd.concat(all_data)

        if add_to_database:
            create_table_and_save_to_sql(
                results.columns, weight_name, self._database_path
            )
        return results.sort_values("event_no").reset_index(drop=True)

    def _make_config(
        self,
        config_outdir: str,
        weight_name: str,
        pisa_config_dict: Optional[Dict] = None,
    ) -> str:
        os.makedirs(config_outdir + "/" + weight_name, exist_ok=True)
        if pisa_config_dict is None:
            # Run on standard settings
            pisa_config_dict = {
                "reco_energy": {"num_bins": 8},
                "reco_coszen": {"num_bins": 8},
                "pid": {"bin_edges": [0, 0.5, 1]},
                "true_energy": {"num_bins": 200},
                "true_coszen": {"num_bins": 200},
                "livetime": 10
                * 0.01,  # set to 1% of 10 years - correspond to the size of the oscNext burn sample
            }

        pisa_config_dict["pipeline"] = self._database_path
        pisa_config_dict["post_fix"] = None
        pipeline_cfg_path = self._create_configs(
            pisa_config_dict, config_outdir + "/" + weight_name
        )
        return pipeline_cfg_path

    def _create_configs(self, config_dict: Dict, path: str) -> str:
        # Update binning config
        root = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        if config_dict["post_fix"] is not None:
            config_name = "config%s" % config_dict["post_fix"]
        else:
            # config_dict["post_fix"] = '_pred'
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
            + "/resources/configuration_templates/pipeline_config_weight_template.cfg",
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


class ContourFitter:
    """Class for fitting contours using PISA."""

    def __init__(
        self,
        outdir: str,
        pipeline_path: str,
        post_fix: str = "_pred",
        model_name: str = "gnn",
        include_retro: bool = True,
        statistical_fit: bool = False,
    ):
        """Construct `ContourFitter`."""
        self._outdir = outdir
        self._pipeline_path = pipeline_path
        self._post_fix = post_fix
        self._model_name = model_name
        self._include_retro = include_retro
        self._statistical_fit = str(statistical_fit)
        self._allowed_contour_types = ["1d", "2d"]

    def fit_1d_contour(
        self,
        run_name: str,
        config_dict: Dict,
        grid_size: int = 30,
        n_workers: int = 1,
        theta23_minmax: Tuple[float, float] = (36.0, 54.0),
        dm31_minmax: Tuple[float, float] = (2.3, 2.7),
    ) -> None:
        """Fit 1D contours."""
        self._fit_contours(
            config_dict=config_dict,
            run_name=run_name,
            grid_size=grid_size,
            n_workers=n_workers,
            theta23_minmax=theta23_minmax,
            dm31_minmax=dm31_minmax,
            contour_type="1d",
        )

    def fit_2d_contour(
        self,
        run_name: str,
        config_dict: Dict,
        grid_size: int = 30,
        n_workers: int = 1,
        theta23_minmax: Tuple[float, float] = (36.0, 54.0),
        dm31_minmax: Tuple[float, float] = (2.3, 2.7),
    ) -> None:
        """Fit 2D contours."""
        self._fit_contours(
            config_dict=config_dict,
            run_name=run_name,
            grid_size=grid_size,
            n_workers=n_workers,
            theta23_minmax=theta23_minmax,
            dm31_minmax=dm31_minmax,
            contour_type="2d",
        )

    def _check_inputs(
        self,
        contour_type: str,
        dm31_minmax: Tuple[float, float],
        theta23_minmax: Tuple[float, float],
        n_workers: int,
    ) -> bool:
        """Check whether inputs are as expected."""
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
        run_name: str,
        config_dict: Dict,
        grid_size: int,
        n_workers: int,
        contour_type: str,
        theta23_minmax: Tuple[float, float],
        dm31_minmax: Tuple[float, float],
    ) -> None:
        """Fit contours."""
        inputs_ok = self._check_inputs(
            contour_type=contour_type,
            dm31_minmax=dm31_minmax,
            theta23_minmax=theta23_minmax,
            n_workers=n_workers,
        )
        if not inputs_ok:
            return

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

    def _merge_temporary_files(self, run_name: str) -> pd.DataFrame:
        files = os.listdir(self._outdir + "/" + run_name + "/tmp")
        df = pd.concat(
            [
                pd.read_csv(f"{self._outdir}/{run_name}/tmp/{file}")
                for file in files
            ],
            ignore_index=True,
        )
        return df

    def _parallel_fit_2d_contour(self, settings: List[List[Any]]) -> None:
        """Fit 2D contours in parallel.

        Length of settings determines the amount of jobs this worker gets.
        Results are saved to temporary .csv-files that are later merged.

        Args:
            settings: A list of fitting settings.
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

    def _parallel_fit_1d_contour(self, settings: List[List[Any]]) -> None:
        """Fit 1D contours in parallel.

        Length of settings determines the amount of jobs this worker gets.
        Results are saved to temporary .csv-files that are later merged.

        Args:
            settings: A list of fitting settings.
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

    def _save_temporary_results(
        self, outdir: str, run_name: str, results: List[List[Any]], id: int
    ) -> None:
        os.makedirs(outdir + "/" + run_name + "/tmp", exist_ok=True)
        results_df = pd.DataFrame(
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
        results_df.to_csv(
            outdir + "/" + run_name + "/tmp" + "/tmp_%s.csv" % id
        )

    def _make_2d_settings(
        self,
        cfgs: Dict,
        grid_size: int,
        run_name: str,
        minimizer_cfg: str,
        theta23_range: Tuple[float, float],
        dm31_range: Tuple[float, float],
        n_workers: int,
    ) -> List[np.ndarray]:
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
        cfgs: Dict,
        grid_size: int,
        run_name: str,
        minimizer_cfg: str,
        theta23_range: Tuple[float, float],
        dm31_range: Tuple[float, float],
        n_workers: int,
    ) -> List[np.ndarray]:
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

    def _setup_config_files(self, run_name: str, config_dict: Dict) -> Dict:
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

    def _get_minimizer_path(self, config_dict: Optional[Dict]) -> str:
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

    def _create_configs(self, config_dict: Dict, path: str) -> str:
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
        outdir: str,
        run_name: str,
        is_retro: bool,
        pipeline_path: str,
        post_fix: str = "_pred",
        config_dict: Optional[Dict] = None,
    ) -> str:
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
