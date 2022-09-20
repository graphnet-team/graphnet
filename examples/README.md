<details>
<summary><b>convert_i3_to_sqlite.py</b></summary>

The script will look in the provided filepath for both the i3 files and the gcd file.

## main_icecube86()
used for the original 86 strings

### variables
`paths` : Can be either a string or list of strings for paths to i3 formated files.
`pulsemap` : the relevant pulsemap of the i3 file.
`gcd_rescue` : Can be either a string of the path to the gcd file if not in same directory as `paths` or `None` if the geometry is contained within the i3 file.
`outdir` : path for the output.

## main_icecube_upgrade()
Used for icecube upgrade

### variables
`basedir`: Can be either a string or list of strings for paths to i3 formated files.
`paths` : String of the filename within the dir 
`pulsemap` : the relevant pulsemap of the i3 file.
`gcd_rescue` : Can be either a string of the path to the gcd file if not in same directory as `paths` or `None` if the geometry is contained within the i3 file.
`outdir` : path for the output.
`workers`: number of cpus to use.