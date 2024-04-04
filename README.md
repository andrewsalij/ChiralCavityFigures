# ChiralCavityFigures
Figures and Data for Chiral Cavity Paper

Datasets and Figures for A 2D Chiral Microcavity based on Apparent Circular Dichroism
Authors
Tzu-Ling Chen,Andrew Salij, Katherine A. Parrish,  Julia K. Rasch, Francesco Zinna, 
Paige J. Brown, Gennaro Pescitelli, Francesco Urraci, Laura A. Aronica, Abitha Dhavamani,
Michael S. Arnold, Michael R. Wasielewski, Lorenzo di Bari,  Roel Tempelaar,  Randall H.
Goldsmith 
and associated Supporting Information 

This repository is available on Zenodo at the following link.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10855107.svg)](https://doi.org/10.5281/zenodo.10855107)

Dependencies made be installed by running
```bash
pip install numpy matplotlib pandas pymatgen scipy sympy 
```
A sample figure is shown here:
<figure>
<img src=https://github.com/andrewsalij/ChiralCavityFigures/blob/main/Figures/440_nm_field_distributions_cavity_21_deg.png alt="Cavity field distributions at 21 degree incidence" width = "800px"/>
<figcaption align = "center">Calculated cavity field distributions for oblique (21 degree) light incidence.</figcaption>
</figure>

This is largely a clone of SalijPhDWork (GPL 3.0) with primarily the plotting routines for PTPO-containing cavities. Plotting routines have been rewritten to be agnostic to local environment provided that the organizational structure of this repository is maintained. 

For navigation, focus on the Figures folder and the folders with raw experimental data. There are many files in the main directory for various plotting routines and last-minute calculations. They have been kept to retain a comprehensive environment. Figures/Source_Files contains high resolution pdfs for figures and raw data tables.

For the Berreman Matrix Method, it relies on an updated version of Pyllama (GPL 3.0) in pyllama.py and cholesteric.py: Bay, M. M., Vignolini, S., & Vynck, K. (2022). PyLlama: A stable and versatile Python toolkit for the electromagnetic modelling of multilayered anisotropic media. Computer Physics Communications, 273, 108256. https://pyllama.readthedocs.io/en/latest/ Pyllama has been updated to allow for spectral dispersion and for modification of the magneto-optic and permeability tensors as well as field visualization. 
