"""Detector-specific modules, for data ingestion and standardisation."""

from .icecube import IceCube86, IceCubeDeepCore, IceCubeUpgrade
from .detector import Detector
from .liquido import LiquidO_v1
from .prometheus import ORCA150
from .magic import MAGIC
from .nubench import FlowerS, FlowerL, FlowerXL, Triangle, Cluster, Hexagon
