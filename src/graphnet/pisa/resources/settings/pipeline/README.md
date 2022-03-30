# Pipeline Settings

This directory contains example pipeline configuration files for most of the stages and the services that implement each stage in PISA.


## Directory listing

| File/directory              | Description
| --------------------------- | -----------
| `example.cfg`               | simplest MC pipeline with no smoothing
| `example_aeffsmooth.cfg`    | "staged approach"; upgoing-only events with stages `flux`, `osc`, and `aeff`; smoothing is used in the `aeff` stage
| `example_cake.cfg`          | simplest `hist`-based ("staged-approach") pipeline with no smoothing
| `example_cfx.cfg`           | MC-reweighting unfolding pipeline that uses the `roounfold` service and includes treatment of discrete systematics 
| `example_dummy.cfg`         | Demonstrates a simple pipeline using just the flux/dummy and osc/dummy file
| `example_gen_lvl.cfg`       | simple MC-reweighting pipeline that processes generator-level events
| `example_mceq.cfg`          | Use of MCEq for flux
| `example_muon_sample.cfg`   | MC-reweighting pipeline for muons
| `example_cake_nusquids.cfg` | Use NuQuids for neutrino oscillations
| `example_param.cfg`         | "staged approach" with smoothing provided by parameterization services for effective areas, reco, and PID (here PID is a separate stage)
| `example_pid_stage.cfg`     | simple "staged-approach" but with PID as a separate stage from reco (note that this has been shown to be less accurate than including PID within reco)
| `example_vacuum.cfg`        | simple "staged approach" but oscillations set to those in vacuum (as opposed to oscillations through Earth, as calculated elsewhere)
| `example_vbwkde.cfg`        | "staged approach" with smoothing of reco resolutions (including PID) via bandwidth KDE (VBWKDE, also known as adaptive KDE)
| `example_xsec.cfg`          | "staged approach" but replacing the `aeff` stage with `xsec`; no reco or PID stages are used
