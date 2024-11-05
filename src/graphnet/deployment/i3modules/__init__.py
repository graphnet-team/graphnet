"""Collection of I3Modules.

This module contains various I3Modules which allow for running GNN model
inference as part of an icetray reconstruction chain, for different IceCube
detector configurations.
"""

from .deprecated_methods import *  # noqa: F403
from graphnet.deployment.icecube import I3InferenceModule, I3PulseCleanerModule
