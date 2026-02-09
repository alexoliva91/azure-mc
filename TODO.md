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

- Since these extra parameters are not explicitly supported by AZURE2, they will not be included in the `parameters.yaml` file created by the `azure-mc params -i input.azr` command, therefore:

    - [ ] Create a middleware layer that takes care of handling these extra parameters included in the 'parameters.yaml' file, ensuring that they are properly applied to the input files used for the AZURE2 calculations during the MCMC sampling process. This middleware layer will be responsible for modifying the input files as needed to reflect the current values of these extra parameters during each iteration of the MCMC sampling process.
    - [ ] Create a new interface for the user to specify these extra parameters in a clear and intuitive way. This interface could be implemented as a command-line option or as a section in the `parameters.yaml` file, depending on what is more convenient for the user.
    - [ ] Rework `parameters.yaml` file structure to accommodate both the parameters explicitly supported by AZURE2 and the extra parameters, ensuring that the file remains organized and easy to understand for the user. For this purpose, we should consider grouping the parameters into different sections, such as "level parameters", "section parameters", to clearly distinguish between the types of parameters and make it easier for the user to navigate the file.


## Medium priority
- Replace ambiguous `populate` command:
  - [ ] Use `azure-mc params -i input.azr` to create only `parameters.yaml`.
  - [ ] Use `azure-mc mc setup` or `azure-mc mcmc setup` to create `mc_setup.yaml` and `mcmc_setup.yaml`, respectively.

- Implement simple plot drawing functionality from *_results.npz files:
  - [ ] Plot posterior distributions for parameters using 'corner.py' or similar library.
  - [ ] Plot the convergence of MCMC chains over iterations to assess mixing and convergence.
  - [ ] Plot the ln(Probability) values over iterations to visualize the likelihood landscape and identify potential issues with the sampling process.
  