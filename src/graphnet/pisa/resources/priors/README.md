# Non-Standard Priors

Typically, we include priors on the nuisance parameters in our fits by adding a penalty term according to a gaussian with a width of some 1 sigma uncertainty on the parameter. In the case of non-standard priors, we instead spline a penalty surface and then just evaluate this at the parameter value.

Here, we only currently use non-standard priors on the atmospheric oscillations mixing angle, &#952;<sub>23</sub>, taken from the [Nu-Fit](http://www.nu-fit.org/) global fits. The naming convention of these files is as follows:

1. All files should start with `nufit`, since in the future there may be a want to take prior surfaces from other global fits, and this will aid in distinguishing these.
2. The version is then listed. This is done without the decimal point and prefixed with a `v`. So version 2.0 is `v20`.
3. Exactly which penalty surface has been splined is then written here. This will be explained after the generic naming convention.
4. The variable that the penalty surface is for is then written next. All of these should say `theta23`.
5. Lastly, the file ends with either `spline` or `splines`. The plural or singular is important here since it tells us if there are different choices in here. For example, we often have a different penalty depending on which ordering hypothesis is being tested, but not always.

These should all be `json` files since they are python dictionary objects that will be loaded in to the `Prior` class (see the PISA core objects for the source code).

For the case of &#952;<sub>23</sub> splines, there are three treatments that have been applied in calculating the penalty surfaces. Since & &#952;<sub>23</sub> is heavily degenerate with the ordering itself it may be desirable to remove any prior information on the ordering that is brought along with the & &#952;<sub>23</sub> prior. Therefore, the three treatments are:

1. `standard` - This refers to just splining the penalty surfaces as given by directly by Nu-Fit. So there is a separate surface for each of the ordering hypotheses and there exists some &#916;&#967;<sup>2</sup> between the respective minima.
2. `shifted` - Again we still have a separate surface for each of the ordering hypotheses, but the disfavoured one has had the &#916;&#967;<sup>2</sup> between the respective minima subtracted from all of the knots in the spline.
3. `minimised` - Since one could believe that a separate penalty surface depending on ordering hypothesis could still be carrying some form of prior information on the ordering, this instead constructs a single spline as the minimum of both of the standard splines across &#952;<sub>23</sub>. Note this is a single spline and this fact is reflected in the filename.

All of these spline surfaces can be re-calculated with the `make_nufit_theta23_spline_priors.py` for any future iterations of Nu-Fit (with the caveat that they do not change their standard data release format). Simply download the `.txt.gz` file they provide with all of their &#916;&#967;<sup>2</sup> surfaces and point this script to them. It is designed to intelligently calculate an appropriate name based on the one in the file.

Also of note are the following two points:

* The provided data from Nu-Fit are the &#916;&#967;<sup>2</sup> surfaces, but since the base metric in PISA is the likelihood, these are divided by 2 and the spline is taken of the negative of the data. If one requests some form of &#967;<sup>2</sup> as the minimisation metric, the appropriate factors will be applied to turn it back in to the correct form.
* In the case of Nu-Fit 2.1 there are two versions of all of the splines labelled `LEM` and `LID`. This is due to the way NOvA released their data for this global fit. Please see the Nu-Fit website and the appropriate NOvA [paper](https://arxiv.org/abs/1601.05022) for more details.