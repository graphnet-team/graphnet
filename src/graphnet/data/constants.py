"""Global constants that are used with `graphnet.data`."""


class FEATURES:
    """Namespace for standard names working with `I3FeatureExtractor`."""

    ICECUBE86 = [
        "dom_x",
        "dom_y",
        "dom_z",
        "dom_time",
        "charge",
        "rde",
        "pmt_area",
    ]
    DEEPCORE = ICECUBE86
    UPGRADE = DEEPCORE + [
        "string",
        "pmt_number",
        "dom_number",
        "pmt_dir_x",
        "pmt_dir_y",
        "pmt_dir_z",
        "dom_type",
    ]


class TRUTH:
    """Namespace for standard names working with `I3TruthExtractor`."""

    ICECUBE86 = [
        "energy",
        "energy_track",
        "position_x",
        "position_y",
        "position_z",
        "azimuth",
        "zenith",
        "pid",
        "elasticity",
        "sim_type",
        "interaction_type",
        "interaction_time",  # Added for vertex reconstruction
        "inelasticity",
        "stopped_muon",
    ]
    DEEPCORE = ICECUBE86
    UPGRADE = DEEPCORE
