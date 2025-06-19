"""Modules for mapping images.

´ImageDefinition´ defines the nodes and the mapping,  and contains general
image-manipulation.´PixelMapping´ defines how raw data is mapped into the
regular sized image.
"""

from .image_definition import ImageDefinition
from .images import IC86Image
from .mappings import IC86PixelMapping
