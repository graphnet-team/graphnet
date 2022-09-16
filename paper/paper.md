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

Optical neutrino telescopes, such as ANTARES [@ANTARES:2011hfw], IceCube [@Aartsen:2016nxy; @DeepCore], KM3NeT [@KM3Net:2016zxf], and Baikal-GVD [@Baikal-GVD:2018isr], detect thousands of particle interaction per seconds, of which neutrino events constitute a miniscule fraction.

The goal of reconstruction at these experiments is to analyse the patterns of light detected in optical modules to infer what particle interaction took place, what the properties of the particles were, etc. The current state-of-the-art reconstruction [@IceCube:2022kff] leverages detailed per-event likelihood optimisation, providing precise reconstruction with inference times of O(30 sec./event).

`GraphNeT` [@graphnet_zenodo:2022] is a python package aimed at providing high quality, user friendly, end-to-end functionality to perform reconstruction tasks in neutrino telescope experiments using graph neural networks (GNNs). `GraphNeT` makes it fast and easy to train complex models that can provide event reconstruction with state-of-the-art performance, for arbitrary detector configurations, with inference times that are orders of magnitude faster than classical reconstruction techniques by separating training and inference [@gnn_icecube].

GNNs from `GraphNeT` are flexible enough to be applicable applied to data from all optical neutrino telescopes, including future projects such as IceCube extensions [@IceCube-PINGU:2014okk; @IceCube:2016xxt;@IceCube-Gen2:2020qha] or P-ONE [@P-ONE:2020ljt].

This means that GNN-based reconstruction can be used to provide state-of-the-art performance on most reconstruction tasks in neutrino telescopes, at O(kHz) event rates, across experiments and physics analyses, with vast potential impact for neutrino physics.


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

Figures can be included like this:

![High-level overview of a typical workflow using `GraphNeT`. Domain-specific data is converted and read using the components in `graphnet.data`. Models are configured, built, trained, and logged using the components in `graphnet.models`. Finally, trained models are deployed to a domain-specific reconstruction chain, yielding predictions, using the components in `graphnet.deployment`.\label{fig:flowchart}](flowchart.pdf)

and referenced from text using \autoref{fig:flowchart}.


# Acknowledgements

Andreas Søgaard has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 890778.


# References