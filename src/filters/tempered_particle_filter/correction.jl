# The outputs of the weight_kernel function are meant to make calculating
# the incremental weight much more efficient to speed up the adaptive φ finding
# that way we're not doing the same matrix multiplication step (dot(p_error, inv_HH*p_error))
# for every iteration of the root-solving algorithm
# Also, the exponential terms are logged first and then exponentiated in the
# incremental_weight calculation so the problem is well-conditioned (i.e. not exponentiating
# very large negative numbers)
function weight_kernel!(coeff_terms::V, log_e_1_terms::V, log_e_2_terms::V,
                        φ_old::Float64, Ψ::Function, y_t::Vector{Float64},
                        s_t_nontemp::AbstractMatrix{Float64},
                        det_HH::Float64, inv_HH::Matrix{Float64};
                        initialize::Bool = false, parallel::Bool = false) where V<:AbstractVector{Float64}
    # Sizes
    n_particles = length(coeff_terms)
    n_obs = length(y_t)

    @mypar parallel for i in 1:n_particles
        error    = y_t - Ψ(s_t_nontemp[:, i])
        sq_error = dot(error, inv_HH * error)

        if initialize
            # Initialization step (using 2π instead of φ_old)
            coeff_terms[i]   = (2*pi)^(-n_obs/2) * det_HH^(-1/2)
            log_e_1_terms[i] = 0.
            log_e_2_terms[i] = -1/2 * sq_error
        else
            # Non-initialization step (tempering and final iteration)
            coeff_terms[i]   = (φ_old)^(-n_obs/2)
            log_e_1_terms[i] = -1/2 * (-φ_old) * sq_error
            log_e_2_terms[i] = -1/2 * sq_error
        end
    end
    return nothing
end

function next_φ(φ_old::Float64, coeff_terms::V, log_e_1_terms::V, log_e_2_terms::V,
                n_obs::Int, r_star::Float64, stage::Int; fixed_sched::Vector{Float64} = Float64[],
                findroot::Function = bisection, xtol::Float64 = 1e-3) where V<:AbstractVector{Float64}

    if isempty(fixed_sched)
        # Solve for optimal φ
        solve_ineff_func(φ) =
            solve_inefficiency(φ, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs) - r_star

        if stage == 1 || (sign(solve_ineff_func(φ_old)) != sign(solve_ineff_func(1.0)))
            # Solve for optimal φ if either
            # 1. First stage
            # 2. Sign change from solve_ineff_fun(φ_old) to solve_ineff_func(1.0)
            return findroot(solve_ineff_func, φ_old, 1.0, xtol = xtol)
        else
            # Otherwise, set φ = 1
            return 1.0
        end
    else
        return fixed_sched[stage]
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