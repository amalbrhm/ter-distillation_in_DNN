# TER – Geometry of Neural Representations in Deep Networks

## Project context

This repository contains the code developed as part of a *Travail d’Étude et de Recherche (TER)*.
The project focuses on the geometric analysis of deep neural networks through the study of
**neural representation trajectories**.

More precisely, the work is inspired by the paper:

> Lange et al. (2023). *Deep Networks as Paths on the Manifold of Neural Representations.*

and aims at understanding, reproducing, and extending some of the experimental results
presented in this work, notably through the use of **Gram matrices** and **Angular CKA**
to compare neural representations across network layers and training epochs.

---

## Objectives

The main objectives of this TER are:

- To study neural representations from a geometric perspective;
- To understand how deep networks can be interpreted as trajectories on a manifold;
- To implement and analyze similarity measures between representations (Gram matrices, CKA);
- To reproduce key experiments from the reference paper;
- To provide a clean and reproducible experimental framework.

---

## Repository structure

```text
.
├── src/                # Core source code (models, metrics, utilities)
├── experiments/        # Scripts to reproduce experiments
├── data/               # Data directory (empty or with download instructions)
├── figures/            # Generated figures and visualizations
├── report/             # LaTeX sources of the TER report
├── README.md
└── requirements.txt
