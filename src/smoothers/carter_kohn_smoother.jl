"""
```
carter_kohn_smoother{S<:AbstractFloat}(m::AbstractModel, df::DataFrame,
    system::System, kal::Kalman{S};
    cond_type::Symbol = :none, include_presample::Bool = false)

carter_kohn_smoother{S<:AbstractFloat}(m::AbstractModel, data::Matrix{S},
    system::System, kal::Kalman{S};
    include_presample::Bool = false)

carter_kohn_smoother{S<:AbstractFloat}(m::AbstractModel, data::Matrix{S},
    T::Matrix{S}, R::Matrix{S}, z0::Vector{S},
    pred::Matrix{S}, vpred::Array{S, 3},
    filt::Matrix{S}, vfilt::Array{S, 3};
    include_presample::Bool = false)
```
This program is a simulation smoother based on Carter and Kohn's
\"On Gibbs Sampling for State Space Modeks\" (Biometrika, 1994).
It recursively sampling from the conditional distribution of time t
states given the full set of observables and states from time t+1 to
time T. Unlike the Durbin Koopman simulation smoother, this one does
rely on inverting potentially singular matrices using the Moore-Penrose
pseudoinverse.

Smoothed shocks are extracted by mapping the forecast errors implied by the
smoothed states back into shocks. As such, this routine assumes that R has
sufficient rank to have a left inverse (i.e. that there are more states than shocks).

### Inputs:

- `m`: model object
- `data`: the (`Ny` x `Nt`) matrix of observable data
- `T`: the (`Nz` x `Nz`) transition matrix
- `R`: the (`Nz` x `Ne`) matrix translating shocks to states
- `z0`: the (`Nz` x 1) initial (time 0) states vector
- `pred`: the (`Nz` x `Nt`) matrix of one-step-ahead predicted states (from the
  Kalman Filter)
- `vpred`: the (`Nz` x `Nz` x `Nt`) matrix of one-step-ahead predicted
  covariance matrices
- `filt`: the (`Nz` x `Nt`) matrix of filtered states
- `vfilt`: the (`Nz` x `Nz` x `Nt`) matrix of filtered covariance matrices
- `cond_type`: optional keyword argument specifying the conditional data type:
  one of `:none`, `:semi`, or `:full`. This is only necessary when a DataFrame
  (as opposed to a data matrix) is passed in, so that `df_to_matrix` knows how
  many periods of data to keep
- `include_presample`: indicates whether or not to return presample periods in
  the returned smoothed states and shocks. Defaults to `false`

Where:

- `Nz`: number of states
- `Ny`: number of observables
- `Ne`: number of shocks
- `Nt`: number of periods for which we have data

### Outputs:

- `α_hat`: the (`Nz` x `Nt`) matrix of smoothed states
- `η_hat`: the (`Ne` x `Nt`) matrix of smoothed shocks

If `n_presample_periods(m)` is nonzero, the `α_hat` and `η_hat` matrices will be
shorter by that number of columns (taken from the beginning).

### Notes

The state space model is defined as follows:
```
y(t) = Z*α(t) + D             (state or transition equation)
α(t+1) = T*α(t) + R*η(t+1)    (measurement or observation equation)
```
"""
function carter_kohn_smoother{S<:AbstractFloat}(regime_indices::Vector{Range{Int64}},
    data::Matrix{S}, TTTs::Vector{Matrix{S}}, RRRs::Vector{Matrix{S}},
    z0::Vector{S}, pred::Matrix{S}, vpred::Array{S, 3},
    filt::Matrix{S}, vfilt::Array{S, 3};
    n_presample_periods::Int = 0, draw_states::Bool = true)

    n_regimes = length(regime_indices)

    # Dimensions
    T  = size(data,    2) # number of periods of data
    Nz = size(TTTs[1], 1) # number of states
    Ne = size(RRRs[1], 2) # number of shocks

    # Smooth the states recursively, starting at t = T-1 and going backwards
    smoothed_states = copy(filt)

    zend = filt[:, T]
    Pend = vfilt[:, :, T]

    smoothed_states[:, T] = if draw_states
        U, eig, _ = svd(Pend)
        zend + U*diagm(sqrt(eig))*randn(Nz)
    else
        zend
    end

    for i = n_regimes:-1:1
        # Get state-space system matrices for this regime
        regime_periods = regime_indices[i]

        # The smoothed state in t = T is the same as the filtered state
        if i == n_regimes
            regime_periods = regime_periods[1:end-1]
        end

        TTT = TTTs[i]

        for t in reverse(regime_periods)
            J = vfilt[:, :, t] * TTT' * pinv(vpred[:, :, t+1])
            μ = filt[:, t] + J*(smoothed_states[:, t+1] - pred[:, t+1])
            Σ = vfilt[:, :, t] - J*TTT*vfilt[:, :, t]

            smoothed_states[:, t] = if draw_states
                U, eig, _ = svd(Σ)
                μ + U*diagm(sqrt(eig))*randn(Nz)
            else
                μ
            end
        end
    end

    # Map the forecast errors implied by the smoothed states back to shocks
    smoothed_shocks = solve_smoothed_shocks(regime_indices, TTTs, RRRs, z0, smoothed_states)

    # Trim the presample if needed
    if n_presample_periods > 0
        mainsample_periods = n_presample_periods+1:T

        smoothed_states = smoothed_states[:, mainsample_periods]
        smoothed_shocks = smoothed_shocks[:, mainsample_periods]
    end

    return smoothed_states, smoothed_shocks
end