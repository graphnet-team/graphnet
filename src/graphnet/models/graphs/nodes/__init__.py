"""Deprecated imports.

moved to graphnet.models.data_representation.graphs.nodes
"""

from graphnet.utilities.logging import Logger


from graphnet.models.data_representation.graphs import (
    NodeDefinition,
    NodesAsPulses,
    PercentileClusters,
    NodeAsDOMTimeSeries,
    IceMixNodes,
)


Logger(log_folder=None).warning_once(
    (
        "`graphnet.models.graphs` will be depricated soon. "
        "All functionality has been moved to "
        "`graphnet.models.data_representation`."
    )
)
