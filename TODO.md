# TODO

## High priority 
- Add the ability to use 'autocorrelation time' as a convergence diagnostic for MCMC chains:
  - [ ] Implement the calculation of autocorrelation time
  - [ ] Use the autocorrelation time to assess the convergence of MCMC chains and determine when to stop sampling.

- Add other R-matrix parameters even if not explicitly supported by AZURE2 (the parameters may be included in the input
file but do not have a corresponding "variable" flag as they are considered constant), such as:
  - [ ] Channel raius (included in the input file but considered constant by AZURE2)
  - [ ] Energy shift of certain experimental data sets
    - Instead of using the experimental data file from the original project directory, create a new file with the same data but with the energy values shifted by the amount specified by the parameter and include this new file in the temporary directory for the calculation. In this way, the energy shift parameter can be varied during the MCMC sampling process and its effect on the fit can be assessed.
  - [ ] Parameters used in the experimental effects calculations, such as the target thickness or the energy resolution of the detector, which can also be included in the input file but are considered constant by AZURE2 (see my fork of BRICK for an example of how to implement this functionality)
    - By allowing these parameters to vary during the MCMC sampling process, it may be possible to better account for systematic uncertainties in the experimental data and improve the overall fit of the model to the data.

## Medium priority
- Replace ambiguous `populate` command:
  - [ ] Use `azure-mc params -i input.azr` to create only `parameters.yaml`.
  - [ ] Use `azure-mc mc setup` or `azure-mc mcmc setup` to create `mc_setup.yaml` and `mcmc_setup.yaml`, respectively.

- Implement simple plot drawing functionality from *_results.npz files:
  - [ ] Plot posterior distributions for parameters using 'corner.py' or similar library.
  - [ ] Plot the convergence of MCMC chains over iterations to assess mixing and convergence.
  - [ ] Plot the ln(Probability) values over iterations to visualize the likelihood landscape and identify potential issues with the sampling process.
  