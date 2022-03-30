# Oscillations Parameter Settings

This directory contains standard settings for the neutrino mixing parameters and parameters that affect the calculation of oscillations.

Typically we have used [Nu-Fit](http://www.nu-fit.org/) to define our fiducial model, and so you should find those here.
The naming convention adopted is that we neglect the decimal point, so nufitv20 here is Nu-Fit v2.0 on the website.
In these files, the priors included on &#952;<sub>23</sub> are the "shifted" ones, as to be consistent with what was done in the LoI V2.
For an explanation of the different &#952;<sub>23</sub> priors please check the `README.md` file in that directory.

An important note on the atmospheric mass splitting values included in the files:
We always use &#916;m<sup>2</sup><sub>31</sub> regardless of ordering, whereas Nu-Fit report &#916;m<sup>2</sup><sub>3l</sub> (that is an ell instead of a one), which is always the bigger of the two mass splittings.
Thus, in order to have the correct value in our configuration files we must add &#916;m<sup>2</sup><sub>21</sub> to the inverted ordering &#916;m<sup>2</sup><sub>31</sub> value from Nu-Fit.
That is, the _absolute_ value will decrease.

## Directory listing

| File/directory | Description
| -------------- | -----------
| `earth.cfg`    | Standard values for the electron densities in the Earth, a standard choice for the propagation height (assumed injection height in the atmosphere of neutrinos when calculating their baseline), and detector depth (appropriate for IceCube)
| `nufitv20.cfg` | Fiducial model from [Nu-Fit v2.0](http://www.nu-fit.org/?q=node/92)
| `nufitv22.cfg` | Fiducial model from [Nu-Fit v2.2](http://www.nu-fit.org/?q=node/123)