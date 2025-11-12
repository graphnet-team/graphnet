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
    PROMETHEUS = [
        "sensor_pos_x",
        "sensor_pos_y",
        "sensor_pos_z",
        "t",
    ]
    KAGGLE = ["x", "y", "z", "time", "charge", "auxiliary"]
    LIQUIDO = ["sipm_x", "sipm_y", "sipm_z", "t"]
    KM3NET = [
        "t",
        "pos_x",
        "pos_y",
        "pos_z",
        "dir_x",
        "dir_y",
        "dir_z",
        "tot",
        "trig",
    ]
    KM3NET_HNL = KM3NET


class TRUTH:
    """Namespace for standard names working with `I3TruthExtractor`."""

    ICECUBE86 = [
        "energy",
        "energy_track",
        "energy_cascade",
        "position_x",
        "position_y",
        "position_z",
        "azimuth",
        "zenith",
        "pid",
        "elasticity",
        "interaction_type",
        "interaction_time",  # Added for vertex reconstruction
        "inelasticity",
        "stopped_muon",
    ]
    DEEPCORE = ICECUBE86
    UPGRADE = DEEPCORE
    PROMETHEUS = [
        "injection_energy",
        "injection_type",
        "injection_interaction_type",
        "injection_zenith",
        "injection_azimuth",
        "injection_bjorkenx",
        "injection_bjorkeny",
        "injection_position_x",
        "injection_position_y",
        "injection_position_z",
        "injection_column_depth",
        "primary_lepton_1_type",
        "primary_hadron_1_type",
        "primary_lepton_1_position_x",
        "primary_lepton_1_position_y",
        "primary_lepton_1_position_z",
        "primary_hadron_1_position_x",
        "primary_hadron_1_position_y",
        "primary_hadron_1_position_z",
        "primary_lepton_1_direction_theta",
        "primary_lepton_1_direction_phi",
        "primary_hadron_1_direction_theta",
        "primary_hadron_1_direction_phi",
        "primary_lepton_1_energy",
        "primary_hadron_1_energy",
        "total_energy",
    ]
    KAGGLE = ["zenith", "azimuth"]
    LIQUIDO = [
        "vertex_x",
        "vertex_y",
        "vertex_z",
        "zenith",
        "azimuth",
        "interaction_time",
        "energy",
        "pid",
    ]
    KM3NET = [
        "true_pdgid",
        "true_E",
        "true_pos_x",
        "true_pos_y",
        "true_pos_z",
        "true_dir_x",
        "true_dir_y",
        "true_dir_z",
        "true_zenith",
        "true_azimuth",
        "run_id",
        "evt_id",
        "frame_index",
        "trigger_counter",
        "n_hits",
        "event_no",
        "is_cc_flag",
        "tau_topology",
    ]
    KM3NET_HNL = [
        "true_pdgid",
        "true_E",
        "true_pos_x",
        "true_pos_y",
        "true_pos_z",
        "true_dir_x",
        "true_dir_y",
        "true_dir_z",
        "true_zenith",
        "run_id",
        "evt_id",
        "frame_index",
        "trigger_counter",
        "n_hits",
        "event_no",
        "is_cc_flag",
        "tau_topology",
        "zenith_hnl",
        "azimuth_hnl",
        "angle_between_showers",
        "Energy_hnl",
        "Energy_second_shower",
        "Energy_imbalance",
        "distance",
    ]
