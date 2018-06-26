function next_φ!(Ψ::Function, stage::Int, φ_old::Float64, det_HH::Float64, inv_HH::Matrix{Float64},
                 y_t::Vector{Float64}, s_t_nontemp::AbstractMatrix{Float64},
                 coeff_terms::V, log_e_1_terms::V, log_e_2_terms::V, r_star::Float64;
                 adaptive::Bool = true, findroot::Function = bisection, xtol::Float64 = 1e-3,
                 fixed_sched::Vector{Float64} = zeros(0),
                 parallel::Bool = false) where V<:AbstractVector{Float64}
    # Sizes
    n_particles = length(coeff_terms)
    n_obs = length(y_t)

    # Compute weight kernel terms
    @mypar parallel for i in 1:n_particles
        p_err = y_t - Ψ(s_t_nontemp[:, i])
        coeff_terms[i], log_e_1_terms[i], log_e_2_terms[i] =
            weight_kernel(φ_old, y_t, p_err, det_HH, inv_HH, initialize = stage == 1)
    end

    # Determine φ_new
    φ_new = if adaptive
        # Compute interval
        solve_ineff_func(φ) =
            solve_inefficiency(φ, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs) - r_star

        if stage == 1
            findroot(solve_ineff_func, φ_old, 1.0, xtol = xtol)
        else
            fphi_interval = [solve_ineff_func(φ_old) solve_ineff_func(1.0)]

            # Look for optimal φ within the interval
            if prod(sign.(fphi_interval)) == -1
                findroot(solve_ineff_func, φ_old, 1.0, xtol = xtol)
            else
                1.0
            end
        end
    else
        fixed_sched[stage]
    end
end

function correction!(φ_new::Float64, coeff_terms::V, log_e_1_terms::V, log_e_2_terms::V,
                     n_obs::Int, inc_weights::Vector{Float64},
                     norm_weights::Vector{Float64}) where V<:AbstractVector{Float64}
    # Compute incremental weights
    n_particles = length(inc_weights)
    for i = 1:n_particles
        inc_weights[i] =
            incremental_weight(φ_new, coeff_terms[i], log_e_1_terms[i], log_e_2_terms[i], n_obs)
    end

    # Normalize weights
    norm_weights .= inc_weights ./ mean(inc_weights)

    return nothing
end

# The outputs of the weight_kernel function are meant to make calculating
# the incremental weight much more efficient to speed up the adaptive φ finding
# that way we're not doing the same matrix multiplication step (dot(p_error, inv_HH*p_error))
# for every iteration of the root-solving algorithm
# Also, the exponential terms are logged first and then exponentiated in the
# incremental_weight calculation so the problem is well-conditioned (i.e. not exponentiating
# very large negative numbers)
function weight_kernel(φ_old::Float64, y_t::Vector{Float64},
                       p_error::Vector{Float64}, det_HH::Float64, inv_HH::Matrix{Float64};
                       initialize::Bool = false)

    sq_error = dot(p_error, inv_HH * p_error)

    if initialize
        # Initialization step (using 2π instead of φ_old)
        coeff_term = (2*pi)^(-length(y_t)/2) * det_HH^(-1/2)
        log_e_term_1   = 0.
        log_e_term_2   = -1/2 * sq_error
    else
        # Non-initialization step (tempering and final iteration)
        coeff_term = (φ_old)^(-length(y_t)/2)
        log_e_term_1   = -1/2 * (-φ_old) * sq_error
        log_e_term_2   = -1/2 * sq_error
    end
    return coeff_term, log_e_term_1, log_e_term_2
end

"""
```
solve_inefficiency{S<:AbstractFloat}(φ_new::S, φ_old::S, y_t::Vector{S}, p_error::Matrix{S},
inv_HH::Matrix{S}, det_HH::S; initialize::Bool=false)
```
Returns the value of the ineffeciency function InEff(φₙ), where:

        InEff(φₙ) = (1/M) ∑ᴹ (W̃ₜʲ(φₙ))²

Where ∑ is over j=1...M particles, and for a particle j:

        W̃ₜʲ(φₙ) = w̃ₜʲ(φₙ) / (1/M) ∑ᴹ w̃ₜʲ(φₙ)

Where ∑ is over j=1...M particles, and incremental weight is:

        w̃ₜʲ(φₙ) = pₙ(yₜ|sₜʲ'ⁿ⁻¹) / pₙ₋₁(yₜ|sₜ^{j,n-1})
                = (φₙ/φₙ₋₁)^(d/2) exp{-1/2 [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]' (φₙ-φₙ₋₁) ∑ᵤ⁻¹ [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]}

### Inputs

- `φ_new`: φₙ
- `φ_old`: φₙ₋₁
- `y_t`: vector of observables for time t
- `p_error`: (`n_states` x `n_particles`) matrix of particles' errors yₜ - Ψ(sₜʲ'ⁿ⁻¹) in columns
- `inv_HH`: The inverse of the measurement error covariance matrix, ∑ᵤ
- `det_HH`: The determinant of the measurement error covariance matrix, ∑ᵤ

### Keyword Arguments

- `initialize::Bool`: flag to indicate whether this is being used in initialization stage,
    in which case one instead solves the formula for w̃ₜʲ(φₙ) as:

    w̃ₜʲ(φ₁) = (φ₁/2π)^(d/2)|∑ᵤ|^(1/2) exp{-1/2 [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]' φ₁ ∑ᵤ⁻¹ [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]}
"""
function solve_inefficiency(φ_new::Float64, coeff_terms::V, exp_1_terms::V, exp_2_terms::V, n_obs::Int64;
                            parallel::Bool = false) where V<:AbstractVector{Float64}
    # Compute incremental weights
    n_particles = length(coeff_terms)
    w = Vector{Float64}(n_particles)
    for i = 1:n_particles
        w[i] = incremental_weight(φ_new, coeff_terms[i], exp_1_terms[i], exp_2_terms[i], n_obs)
    end

    # Compute normalized weights
    W = w/mean(w)

    return sum(W.^2)/n_particles
end

function incremental_weight(φ_new::Float64, coeff_term::Float64, log_e_term_1::Float64,
                            log_e_term_2::Float64, n_obs::Int64)
    return φ_new^(n_obs/2) * coeff_term * exp(log_e_term_1) * exp(φ_new*log_e_term_2)
end