---
title: 'GraphNeT: Graph neural networks for neutrino telescope event reconstruction'

tags:
  - Python
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

Neutrino telescopes, such as ANTARES [@ANTARES:2011hfw], IceCube [@Aartsen:2016nxy; @DeepCore], KM3NeT [@KM3Net:2016zxf], and Baikal-GVD [@Baikal-GVD:2018isr] have the science goal of detecting neutrinos and measuring their properties and origins. Reconstruction at these experiments is concerned with classifying the type of event or estimating properties of the interaction.

`GraphNeT` [@graphnet_zenodo:2022] is an open-source Python framework aimed at providing high quality, user friendly, end-to-end functionality to perform reconstruction tasks at neutrino telescopes using graph neural networks (GNNs). `GraphNeT` makes it fast and easy to train complex models that can provide event reconstruction with state-of-the-art performance, for arbitrary detector configurations, with inference times that are orders of magnitude faster than traditional reconstruction techniques [@gnn_icecube].

GNNs from `GraphNeT` are flexible enough to be applied to data from all neutrino telescopes, including future projects such as IceCube extensions [@IceCube-PINGU:2014okk; @IceCube:2016xxt; @IceCube-Gen2:2020qha] or P-ONE [@P-ONE:2020ljt]. This means that GNN-based reconstruction can be used to provide state-of-the-art performance on most reconstruction tasks in neutrino telescopes, at real-time event rates, across experiments and physics analyses, with vast potential impact for neutrino and astro-particle physics.


# Statement of need

Neutrino telescopes typically consist of thousands of optical modules (OMs) to detect the Cherenkov light produced from particle interactions in the detector medium. The number of photo-electrons recorded by the OMs in each event roughly scales with the energy of the incident particle, from a few photo-electrons and up to tens of thousands.

Reconstructing the particle type and parameters from individual recordings (called events) in these experiments is a challenge due to irregular detector geometry, inhomogeneous detector medium, sparsity of the data, the large variations of the amount of signal between different events, and the sheer number of events that need to be reconstructed.

Multiple approaches have been employed, including relatively simple methods [@Aguilar:2011zz; @IceCube:2022kff] that are robust but limited in precision and likelihood-based methods [@ANTARES:2017ivh; @Ahrens:2003fg; @Aartsen:2013vja; @Abbasi_2013; @Aartsen:2013bfa; @IceCube:2021oqo; @IceCube:2022kff; @Chirkin:2013avz] that can attain a high accuracy at the price of high computational cost and detector specific assumptions.

Recently, machine learning (ML) methods have started to be used, such as convolutional neural networks (CNNs) [@Abbasi:2021ryj; @Aiello:2020orq] that are comparatively fast, but require detector data being transformed into a regular pixel or voxel grid. Other approaches get around the geometric limitations, but increase the computational cost to a similar level as the traditional likelihood methods [@Eller:2022xvi].

Instead, GNNs can be thought of as generalised CNNs that work on data with any geometry, making this paradigm a natural fit for neutrino telescope data.

The `GraphNeT` framework provides the end-to-end tools to train and deploy GNN reconstruction models. `GraphNeT` leverages industry-standard tools such as `pytorch` [@NEURIPS2019_9015], `PyG` [@Fey_Fast_Graph_Representation_2019], `lightning` [@Falcon_PyTorch_Lightning_2019], and `wandb` [@wandb] for building and training GNNs as well as particle physics standard tools such as `awkward` [@jim_pivarski_2020_3952674] for handling the variable-size data representing particle interaction events in neutrino telescopes. The inference speed on a single GPU allows for processing the full online datastream of current neutrino telescopes in real-time.


# Impact on physics

`GraphNeT` provides a common framework for ML developers and physicists that wish to use the state-of-the-art GNN tools in their research. By uniting both user groups, `GraphNeT` aims to increase the longevity and usability of individual code contributions from ML developers by building a general, reusable software package based on software engineering best practices, and lowers the technical threshold for physicists that wish to use the most performant tools for their scientific problems.

The `GraphNeT` models can improve event classification and yield very accurate reconstruction, e.g., for low energy neutrinos observed in IceCube. Here, a GNN implemented in `GraphNeT` was applied to the problem of neutrino oscillations in IceCube, leading to significant improvements in both energy and angular reconstruction in the energy range relevant to oscillation studies [@gnn_icecube]. Furthermore, it was shown that the GNN could improve muon vs. neutrino classification and thereby the efficiency and purity of a neutrino sample for such an analysis.

Similarly, improved angular reconstruction has a great impact on, e.g., neutrino point source analyses.

Finally, the fast (order millisecond) reconstruction allows for a whole new type of cosmic alerts at lower energies, which were previously unfeasible. GNN-based reconstruction makes it possible to identify low energy (< 10 TeV) neutrinos and monitor their rate, direction, and energy in real-time. This will enable cosmic neutrino alerts based on such neutrinos for the first time ever, despite a large background of neutrinos that are not of cosmic origin.


# Usage

`GraphNeT` comprises a number of modules providing the necessary tools to build workflows from ingesting raw training data in domain-specific formats to deploying trained models in domain-specific reconstruction chains, as illustrated in \autoref{fig:flowchart}.

![High-level overview of a typical workflow using `GraphNeT`: `graphnet.data` enables converting domain-specific data to industry-standard, intermediate file formats and reading this data; `graphnet.models` allows for configuring and building complex GNN models using simple, physics-oriented components; `graphnet.training` manages model training and experiment logging; and finally, `graphnet.deployment` allows for using trained models for inference in domain-specific reconstruction chains.\label{fig:flowchart}](flowchart.pdf)

`graphnet.models` provides modular components subclassing `torch.nn.Module`, meaning that users only need to import a few existing, purpose-built components and chain them together to form a complete GNN. ML developers can contribute to `GraphNeT` by extending this suite of model components — through new layer types, physics tasks, graph connectivities, etc. — and experiment with optimising these for different reconstruction tasks using experiment tracking.

These models are trained using `graphnet.training` on data prepared using `graphnet.data`, to satisfy the high I/O loads required when training ML models on large batches of events, which domain-specific neutrino physics data formats typically do not allow.

Trained models are deployed to a domain-specific reconstruction chain, yielding predictions, using the components in `graphnet.deployment`. This can either be through model files or container images, making deployment as portable and dependency-free as possible.

By splitting up the GNN development as in \autoref{fig:flowchart}, `GraphNeT` allows physics users to interface only with high-level building blocks or pre-trained models that can be used directly in their reconstruction chains, while allowing ML developers to continuously improve and expand the framework’s capabilities.


# Acknowledgements

Andreas Søgaard has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 890778.
The work of Rasmus Ørsøe was partly performed in the framework of the PUNCH4NFDI consortium supported by DFG fund "NFDI 39/1", Germany.


# References
