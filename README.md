
# Data-Augmentation

## Overview
This repository contains all code and resources for the experiments presented in our paper:

**"Addressing data security in iot: Minimum sample size and denoising diffusion models for improved malware detection"**  
*(Link to the paper: [Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=CMPBKJkAAAAJ&citation_for_view=CMPBKJkAAAAJ:zYLM7Y9cAGgC))*

The work focuses on generating robust augmented data to improve model generalization in scenarios with limited, noisy, or distributed datasets. The repository supports experiments in image classification, IoT sensor data, and other structured datasets.

## Research Context
The project is highly relevant to **IoT, edge AI, and distributed learning**, where data may be sparse or privacy-sensitive. Data augmentation allows models to generalize better without requiring extensive centralized data collection, contributing also to **cybersecurity and privacy-preserving machine learning**.

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    DATA AUGMENTATION PIPELINE                   │
└─────────────────────────────────────────────────────────────────┘

                              INPUT DATA
                                  ↓
                    ┌─────────────────────────┐
                    │  Training/Test Dataset  │
                    │  (Grayscale Images)     │
                    └─────────────────────────┘
                                  ↓
            ┌─────────────────────┬─────────────────────┐
            ↓                     ↓                     ↓
    ┌───────────────┐  ┌──────────────────┐  ┌────────────────┐
    │ CLASSIFICATION│  │  GENERATION_DIFF │  │ GENERATION_GAN │
    └───────────────┘  └──────────────────┘  └────────────────┘
```
Data Flow
```text

         ┌─────────────────────────────────────┐
         │    DDPM-Generated Synthetic Data    │
         │    + GAN-Generated Synthetic Data   │
         │    + Original Real Data             │
         └────────────────┬────────────────────┘
                          ↓
              ┌────────────────────────┐
              │  COMBINED DATASET      │
              │  (Balanced Training)   │
              └────────────────┬───────┘
                               ↓
              ┌────────────────────────┐
              │  Classification Model  │
              │  Training & Validation │
              └────────────────┬───────┘
                               ↓
              ┌────────────────────────┐
              │   Inference & Testing  │
              │  F1 Score, ROC Curves  │
              └────────────────────────┘
```


# Key Components
Classification: CNN-based classifier for malware
Diffusion: DDPM for noise-based image generation
GAN: Adversarial network for synthetic data
Utilities: Data loading, balancing, validation
Models: Pre-trained weights per class

# Results
The experiments reported in the paper demonstrate that robust data augmentation improves classification accuracy and generalization across different tasks and datasets. Detailed results, plots, and metrics are included in the project presentation files.

Reproducibility
To reproduce the results from the paper:

Clone the repository:

```bash

git clone https://github.com/ThisPlatypus/Data-Augmentation.git
cd Data-Augmentation

```
#Install required Python packages:

```bash
pip install -r requirements.txt
```


