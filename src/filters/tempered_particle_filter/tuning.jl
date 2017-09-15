"""
```
Tuning{S<:AbstractFloat}
```
### Fields:
- `r_star`: The target ratio such that the chosen φ* satisfies r_star = InEff(φ*) = Sample mean with respect
to the number of particles of the squared, normalized particle weights, W_t^{j,n}(φ_n)^2.
- `c`: The adaptively chosen step size of each proposed move in the mutation step of the tempering iterations portion of
the algorithm.
- `accept_rate`: The rate of the number of particles accepted in the mutation step at each time step, which factors
into the calculation of the adaptively chosen c step.
- `target`: The target acceptance rate, which factors into the calculation of the adaptively chosen c step.
- `xtol`: The error tolerance which the fzero solver function (from the Roots package) uses as a criterion in a sufficiently
accurate root.
- `resampling_method`: The method for resampling particles each time step
- `N_MH`: The number of metropolis hastings steps that are proposed in the mutation step of the tempering iterations portion of
the algorithm.
- `n_particles`: The number of particles that are used to make the log-likelihood approximation (more giving a more accurate
estimate of the log-likelihood at the cost of being more computationally intensive).
- `n_presample_periods`: If greater than 0, the first `n_presample_periods` will be omitted from the likelihood calculation.
- `adaptive`: Whether or not to adaptively solve for an optimal φ schedule w/ resp. to r_star, and instead use the pre-allocated
fixed schedule inputted directly into the tpf function.
- `allout`: Whether or not to return all outputs (log-likelihood, incremental likelihood, and time for each time step iteration)
- `parallel`: Whether or not to run the algorithm with parallelized mutation and resampling steps.
"""
type Tuning{S<:AbstractFloat}
    r_star::S                    # Initial target ratio for φ
    c::S                         # Step size in the mutation step
    accept_rate::S               # Initial average acceptance rate for new particles during mutation
    target::S                    # Initial target acceptance rate for new particles during mutation
    xtol::S                      # The error tolerance for the fzero solver
    resampling_method::Symbol
    N_MH::Int                    # The number of metropolis hastings steps taken during mutation
    n_particles::Int             # The number of particles
    n_presample_periods::Int     # Number of periods before the main sample
    adaptive::Bool               # Whether or not to adaptively solve for an optimal φ schedule
    allout::Bool                 # Whether or not to return all outputs
    parallel::Bool
end

function Tuning(;default_values = false)
    return Tuning(2., 0.3, 0.4, 0.4, 0., :systematic, 1, 1000, 0, true, true, false)
end

@inline function Base.getindex(T::Tuning, d::Symbol)
    if d in (:r_star, :c, :accept_rate, :target, :xtol, :resampling_method, :N_MH, :n_particles,
             :n_presample_periods, :adaptive, :allout, :parallel)
        return getfield(T, d)
    else
        throw(KeyError(d))
    end
end
