"""Global constants that are used across modules."""

class FEATURES:
    ICECUBE86 = [
        'dom_x',
        'dom_y',
        'dom_z',
        'dom_time',
        'charge',
        'rde',
        'pmt_area',
    ]
    DEEPCORE = ICECUBE86
    UPGRADE = DEEPCORE + [
        'string',
        'pmt_number',
        'dom_number',
        'pmt_dir_x',
        'pmt_dir_y',
        'pmt_dir_z',
        'dom_type',
    ]

class TRUTH:
    ICECUBE86 = [
        'energy',
        'position_x',
        'position_y',
        'position_z',
        'azimuth',
        'zenith',
        'pid',
        'elasticity',
        'sim_type',
        'interaction_type',
    ]
    DEEPCORE = ICECUBE86
    UPGRADE = DEEPCORE
