---
title: 'GraphNeT: Graph neural networks for neutrino telescope event reconstruction'

tags:
  - python
  - machine learning
  - deep learning
  - neural networks
  - graph neural networks
  - astrophysics
  - particle physics
  - neutrinos

authors:
  - name: Andreas Søgaard
    orcid: 0000-0002-0823-056X
    affiliation: 1  # "1, 2" (Multiple affiliations must be quoted)
    corresponding: true
  - name: Rasmus F. Ørsøe
    orcid: 0000-0001-8890-4124
    affiliation: 2
  - name: Leon Bozianu
    affiliation: 1
  - name: Morten Holm
    affiliation: 1
  - name: Kaare Endrup Iversen
    affiliation: 1
  - name: Tim Guggenmos
    affiliation: 2
  - name: Martin Ha Minh
    orcid: 0000-0001-7776-4875
    affiliation: 2
  - name: Philipp Eller
    orcid: 0000-0001-6354-5209
    affiliation: 2
  - name: Troels C. Petersen
    orcid: 0000-0003-0221-3037
    affiliation: 1

affiliations:
 - name: Niels Bohr Institute, University of Copenhagen, Denmark
   index: 1
 - name: Technical University of Munich, Germany
   index: 2

date: 16 September 2022

bibliography: paper.bib

---

# Summary

Optical neutrino telescopes, such as ANTARES [@ANTARES:2011hfw], IceCube [@Aartsen:2016nxy; @DeepCore], KM3NeT [@KM3Net:2016zxf], and Baikal-GVD [@Baikal-GVD:2018isr], detect thousands of particle interaction per seconds, with a science goal of detecting neutrinos and measuring their properties and origins. Reconstruction at these experiments is concerned with classifying the type of event, estimating the properties of the incident particles, etc.

`GraphNeT` [@graphnet_zenodo:2022] is an open-source python framework aimed at providing high quality, user friendly, end-to-end functionality to perform reconstruction tasks in neutrino telescope experiments using graph neural networks (GNNs). `GraphNeT` makes it fast and easy to train complex models that can provide event reconstruction with state-of-the-art performance, for arbitrary detector configurations, with inference times that are orders of magnitude faster than traditional reconstruction techniques by separating training and inference [@gnn_icecube].

GNNs from `GraphNeT` are flexible enough to be applicable applied to data from all optical neutrino telescopes, including future projects such as IceCube extensions [@IceCube-PINGU:2014okk; @IceCube:2016xxt; @IceCube-Gen2:2020qha] or P-ONE [@P-ONE:2020ljt]. This means that GNN-based reconstruction can be used to provide state-of-the-art performance on most reconstruction tasks in neutrino telescopes, at near–real-time event rates, across experiments and physics analyses, with vast potential impact for neutrino physics.


# Statement of need

`GraphNeT` is (...)

* GraphNet Github Zenodo [@graphnet_zenodo:2022],
* RETRO [@IceCube:2022kff]
* DNN reco. [@Abbasi:2021ryj]
* Review "Graph neural networks in particle physics" [@Shlomi_2021]
* `pytorch` [@NEURIPS2019_9015]
* `pytorch-geometric` [@Fey_Fast_Graph_Representation_2019]
* Detectors [@Aartsen:2016nxy; @DeepCore; @IceCube-Gen2:2020qha; @IceCube:2016xxt; @ANTARES:2011hfw; @KM3Net:2016zxf; @Baikal-GVD:2018isr; @P-ONE:2020ljt]


# Usage

`GraphNeT` provides a number of modules providing the necessary tools to build workflow from ingesting raw training data in domain-specific formats to deploying trained models in domain-specific reconstruction chains, as illustrated in \autoref{fig:flowchart}.

![High-level overview of a typical workflow using `GraphNeT`.\label{fig:flowchart}](flowchart.pdf)

Domain-specific data is converted and read using the components in `graphnet.data`. This is necessary because neutrino telescope data is often stored in domain-specific file format that is often not suitable for the high I/O loads required when training machine learning (ML) models on large batches of events.

Models are configured, built, trained, and logged using the components in `graphnet.models`. This module contains modular components subclassing `torch.nn.Module`, meaning that users only need to import a few, existing, purpose-built components and chain them together to form a complete GNN. ML developers can contribute to `GraphNeT` by extending this suite of model components — through new layer types, physics tasks, graph connectivities, etc. — and experiment with optimising these for different reconstruction tasks using experiment tracking.

Finally, trained models are deployed to a domain-specific reconstruction chain, yielding predictions, using the components in `graphnet.deployment`. This can either be through model artefacts or container images, making deployment as portable and dependency-free as possible.

By splitting up the GNN development in this way, `GraphNeT` allows physics users to interface only with high-level building block or pre-trained models that can be used directly in their reconstruction chains, while allowing ML developers to continuously improve and expand the framework’s capabilities.



# Acknowledgements

Andreas Søgaard has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 890778.


# References