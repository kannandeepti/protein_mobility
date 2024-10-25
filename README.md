# Simulations of multivalent protein spheres

This folder contains scripts to interface with the open2c/polychrom simulation software package in order to run MD simulations
of multivalent protein spheres. To use these scripts, first create a clean conda environment and install openMM and other dependencies using the install.sh script.
```
conda create -n polychrom python=3.9
conda activate polychrom
bash install.sh
```

The `utils` folder contains custom force fields for creating patched spheres, where patches attract one another and spheres repel one another. The `simulation_scripts` folder contains the scripts used to run polychrom simulations. 
The `analysis_scripts` folder contains scripts for analyzing the output of polychrom simulations, including calculating
mean squared displacements of proteins as a function of time and computing the cluster size distribution.

