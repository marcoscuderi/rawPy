"""
rawPy is a package intended to facilitate data analysis of experimental data.
It furnish a variety of functions that allow to elaborate a raw data file. 
It is also integrated with PyRSF (see below) supported by a GUI to allow for analysis of velocity stepping tests.
Author: MArco M scuderi
Co-authors: Martijn Van den Ende and John Leeman


PyRSF is a light-weight rate-and-state friction (RSF) modelling package.
It features:

- Forward RSF modelling with multiple state parameters
- Inversion of the RSF parameters
- A modified RSF formulation with cut-off velocity
- User-defined state evolution functions
- Stick-slip simulations with stable limit cycles for K < Kc

The main API for this package is pyrsf.inversion.rsf_inversion

Author: Martijn van den Ende (ORCID:0000-0002-0634-7078)
Repository: https://github.com/martijnende/pyrsf
Modification date: 02 April 2018
"""