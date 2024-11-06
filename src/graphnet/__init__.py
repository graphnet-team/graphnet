"""graphnet: Graph neural networks for neutrino telescope event reconstruction.

**graphnet** is a python package that provides convenient, common, and
collaboratively developed tools for building graph neural networks (GNNs) to
solve physics tasks at neutrino telescope experiments. It aims to provide
physicists with the tools to leverage advanced machine learning (ML) without
having to be machine learning experts themselves, and thereby accelerate the
scientific advances in the area of neutrino phyics.

Design principles:

- End-to-end: graphnet aims to provide all of the tools for streamlining the
  process of ingesting and transforming physics data; building, training, and
  optimising GNN models; and deploying them into a reconstruction chain.
- Extensibility: graphnet aims to provide the basic building blocks to improve
  reconstruction and classification across the various IceCube configurations,
  but all model components can be easily extended to new experiments, with new
  GNN architectures and for new physics tasks.

Main features:

- Converters from domain-specific data formats (I3) to more common, indexable
  formats (e.g., SQLite) suitable as intermediate file formats for training ML
  models
- Plug-and-play GNN model components that abstract away ML implementation
  details and only expose the "building blocks" that are most relevant to
  physicsts (e.g., what detector is used and what are the physicst tasks).
- I3Modules for easily including GNN models in IceCube reconstruction chains.
- Docker images for running model inference in a containerised fashion.

"""

from . import _version

__version__ = _version.get_versions()[  # type: ignore[no-untyped-call]
    "version"
]
