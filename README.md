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
## Dataset 

we will be using the famous CIFAR-10 dataset, which has become one of the the most common choice for beginner computer vision datasets. The dataset is a labeled subset of the 80 million tiny images dataset. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. “Automobile” includes sedans, SUVs, and things of that sort. “Truck” includes only big trucks. Neither includes pickup trucks.
## Objectives

The main objectives of this TER are:

- To study neural representations from a geometric perspective;
- To understand how deep networks can be interpreted as trajectories on a manifold;
- To implement and analyze similarity measures between representations (Gram matrices, CKA);
- To reproduce key experiments from the reference paper;
- To provide a clean and reproducible experimental framework.
- ... to be defined further as the project progresses.
---

## Repository structure

