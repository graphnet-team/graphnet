"""Modules for mapping images.

´ImageDefinition´ defines the nodes and the mapping,  and contains general
image-manipulation.´PixelMapping´ defines how raw data is mapped into the
regular sized image.
"""

from .pixel_mappings import (
    PixelMapping,
    IC86DNNMapping,
)
