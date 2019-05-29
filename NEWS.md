# StateSpaceRoutines.jl v0.2.0 Release Notes

## New features and enhancements
- Add option to specify a likelihood convergence tolerance (the keyword argument, `tol`) for the `kalman_filter` and `chand_recursion`.
- Save the `PZV` matrix in the `KalmanFilter` type, (`P_pred' * Z' * inv(V_pred)`), for state and state variance-covariance matrix updating without needing to recompute that matrix product.
- Write an additional `kalman_likelihood` set of methods for obtaining only the likelihood from the kalman filter. While `kalman_filter` can already do so, `kalman_likelihood` is optimized for performance, as it does not instantiate empty return values.

# StateSpaceRoutines.jl v0.1.1 Release Notes

## Performance changes

- `kalman_filter` runs about 20% more quickly
- `hamilton_smoother` and `carter_kohn_smoother` run about 5% more slowly
- `koopman_smoother` runs about 6% more quickly
- `durbin_koopman_smoother` runs about 13% more quickly

## Breaking changes

- `kalman_filter`
  + Replaced `allout::Bool` keyword argument with `outputs::Vector{Symbol}`. For `allout = false`, now specify `outputs = [:loglh]`
  + Return values have changed to `loglh, s_pred, P_red, s_filt, P_filt, s_0, P_0, s_T, P_T`. Note that `loglh` is the vector of conditional log-likelihoods, not the total log-likelihood. The outputs `yprederror, ystdprederror, rmse, rmsd` are no longer returned but can be calculated from the values that are returned
  + Unlike before, the same number of outputs are returned regardless of what's passed to `outputs`. However, some of the outputs may be empty arrays

# StateSpaceRoutines.jl v0.1.0 Release Notes

## New features

- Add tempered particle filter

## Breaking changes

- Upgrade all code for use with Julia v0.6 or higher
- `kalman_filter` returns an additional output variable, a vector of marginal log-likelihoods
