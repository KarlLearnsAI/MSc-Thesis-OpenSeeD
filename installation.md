# Installation Guide for OpenSeeD

This guide details how to set up OpenSeeD and run your first experiment. The instructions have been tested on Ubuntu 20.04 and Windows 11.

## 1. Prerequisites
- **Operating system:** Linux or Windows
- **Python:** 3.8.20
- **GPU:** CUDA-compatible GPU (tested with CUDA 11.3) and drivers installed
- **Tools:** Git and Conda (from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda)

## 2. Clone the repository
```bash
git clone https://github.com/KarlLearnsAI/MSc-Thesis-OpenSeeD
cd MSc-Thesis-OpenSeeD
```

## 3. Recreate the conda environment
An `environment.yml` file is provided to reproduce the main dependencies and automatically download all required sub‑dependencies.
```bash
conda env create -f environment.yml
conda activate vlmaps6
```

## 4. Download pre-trained checkpoint
Obtain a pre-trained weight file and place it in a convenient directory:
```bash
curl -L https://github.com/IDEA-Research/OpenSeeD/releases/download/openseed/model_state_dict_swint_51.2ap.pt -o checkpoints/model_state_dict_swint_51.2ap.pt
```
On Windows you may download the file via the URL above if `curl` is unavailable.

## 5. Run the OpenSeeD semantic segmentation
Execute all cells inside the full-inmemory-test-openseed.ipynb.
