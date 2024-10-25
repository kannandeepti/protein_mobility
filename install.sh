#!/usr/bin/env bash
# Create a clean conda environment and run the following commands.
conda create -n polychrom python=3.9
source activate polychrom
conda install  -c conda-forge openmm
python -m openmm.testInstallation
conda install -c conda-forge openmmtools

pip install -U git+https://github.com/open2c/cooltools
pip install -U git+https://github.com/open2c/polychrom



