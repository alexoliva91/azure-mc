# TODO

## High priority 
- Add the ability to use 'autocorrelation time' as a convergence diagnostic for MCMC chains:
  - [ ] Implement the calculation of autocorrelation time
  - [ ] Use the autocorrelation time to assess the convergence of MCMC chains and determine when to stop sampling.

## Medium priority
- Replace ambiguous `populate` command:
  - [ ] Use `azure-mc params -i input.azr` to create only `parameters.yaml`.
  - [ ] Use `azure-mc mc setup` or `azure-mc mcmc setup` to create `mc_setup.yaml` and `mcmc_setup.yaml`, respectively.

- Implement simple plot drawing functionality from *_results.npz files:
  - [ ] Plot posterior distributions for parameters using 'corner.py' or similar library.
  - [ ] Plot the convergence of MCMC chains over iterations to assess mixing and convergence.
  - [ ] Plot the ln(Probability) values over iterations to visualize the likelihood landscape and identify potential issues with the sampling process.
  