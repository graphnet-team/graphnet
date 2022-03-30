# pisa\_examples.resources

Example PISA resources

To use resources outside of these examples, create a directory outside of the PISA sourcecode and set the `PISA_RESOURCES` environment variable in your shell to point to that directory.


## Directory Listing

| File/directory    | Description
| ----------------- | -----------
| `aeff/`           | Data files that define effective areas
| `cross_sections/` | Detector medium neutrino
| `events/`         | PISA events HDF5 files
| `flux/`           | Neutrino fluxes, e.g. the tables from Honda et al. (2015)
| `osc/`            | The Earth density files (PREM) used for computing neutrino oscillations
| `pid/`            | Particle identification (PID) data files, e.g. parametersied PID probabilities.
| `priors/`         | Pre-defined priors that can be used in analyses, e.g. from nu-fit.org
| `reco/`           | Reconstruction resolutions, e.g. parameterised reconstruction kernels
| `settings/`       | Contains directories, each for settings for different purposes
| `__init__.py`     | File that makes `resources` directory behave as a Python module