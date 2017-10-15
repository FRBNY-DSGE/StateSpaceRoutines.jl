# StateSpaceRoutines.jl v0.1.0 Release Notes

## New features

- Add tempered particle filter

## Breaking changes

- Upgrade all code for use with Julia v0.6 or higher
- `kalman_filter` returns an additional output variable, a vector of marginal log-likelihoods
