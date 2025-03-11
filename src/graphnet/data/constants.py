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
    SNOWSTORM = [
        "dom_x",
        "dom_y",
        "dom_z",
        "charge",
        "dom_time",
        "width",
        "pmt_area",
        "rde",
        "is_bright_dom",
        "is_bad_dom",
        "is_saturated_dom",
        "is_errata_dom",
        "event_time",
        "hlc",
        "awtd",
        "string",
        "pmt_number",
        "dom_number",
        "dom_type",
    ]
    KAGGLE = ["x", "y", "z", "time", "charge", "auxiliary"]
    LIQUIDO = ["sipm_x", "sipm_y", "sipm_z", "t"]


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
        "visible_inelasticity",
        "visible_energy",
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
    SNOWSTORM = [
        "energy",
        "position_x",
        "position_y",
        "position_z",
        "azimuth",
        "zenith",
        "pid",
        "event_time",
        "interaction_type",
        "elasticity",
        "RunID",
        "SubrunID",
        "EventID",
        "SubEventID",
        "dbang_decay_length",
        "track_length",
        "stopped_muon",
        "energy_track",
        "energy_cascade",
        "inelasticity",
        "DeepCoreFilter_13",
        "CascadeFilter_13",
        "MuonFilter_13",
        "OnlineL2Filter_17",
        "L3_oscNext_bool",
        "L4_oscNext_bool",
        "L5_oscNext_bool",
        "L6_oscNext_bool",
        "L7_oscNext_bool",
        "Homogenized_QTot",
        "MCLabelClassification",
        "MCLabelCoincidentMuons",
        "MCLabelBgMuonMCPE",
        "MCLabelBgMuonMCPECharge",
        "GNLabelTrackEnergyDeposited",
        "GNLabelTrackEnergyOnEntrance",
        "GNLabelTrackEnergyOnEntrancePrimary",
        "GNLabelTrackEnergyDepositedPrimary",
        "GNLabelEnergyPrimary",
        "GNLabelCascadeEnergyDepositedPrimary",
        "GNLabelCascadeEnergyDeposited",
        "GNLabelEnergyDepositedTotal",
        "GNLabelEnergyDepositedPrimary",
        "GNLabelHighestEInIceParticleIsChild",
        "GNLabelHighestEInIceParticleDistance",
        "GNLabelHighestEInIceParticleEFraction",
        "GNLabelHighestEInIceParticleEOnEntrance",
        "GNLabelHighestEDaughterDistance",
        "GNLabelHighestEDaughterEFraction",
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
