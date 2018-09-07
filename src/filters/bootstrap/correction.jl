"""
```
weight_kernel!(coeff_terms, log_e_terms, Ψ, y_t,
    s_t_nontemp, det_HH, inv_HH; initialize = false, parallel = false)
```

The outputs of the weight_kernel function are meant to speed up the adaptive φ
finding, so that we don't do the same matrix multiplication step in every
iteration of the root-solving algorithm.

The exponential terms are logged first and then exponentiated in the
incremental weight calculation so the problem is well-conditioned (i.e. not
exponentiating very large negative numbers).

This function modifies `coeff_terms` and `log_e_terms`.
"""
function weight_kernel!(coeff_terms::V, log_e_terms::V,
                        Ψ::Function, y_t::Vector{Float64},
                        s_t_nontemp::AbstractMatrix{Float64},
                        det_HH::Float64, inv_HH::Matrix{Float64};
                        parallel::Bool = false) where V<:AbstractVector{Float64}
    # Sizes
    n_particles = length(coeff_terms)
    n_obs = length(y_t)

    @mypar parallel for i in 1:n_particles
        error    = y_t - Ψ(@view(s_t_nontemp[:, i]))
        sq_error = dot(error, inv_HH * error)

        coeff_terms[i]   = (2*pi)^(-n_obs/2) * det_HH^(-1/2)
        log_e_terms[i] = -1/2 * sq_error
    end
    return nothing
end

"""
```
correction!(inc_weights, norm_weights, coeff_terms, log_e_1_terms,
    log_e_2_terms, n_obs)
```
# NOTE:
Compute (and modify in-place) incremental weights w̃ₜʲ and normalized weights W̃ₜʲ:

        w̃ₜʲ(φₙ) = pₙ(yₜ|sₜʲ'ⁿ⁻¹) / pₙ₋₁(yₜ|sₜʲ'ⁿ⁻¹)
                = (φₙ/φₙ₋₁)^(d/2) exp{-1/2 (φₙ-φₙ₋₁) [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]' Σᵤ⁻¹ [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]}

        W̃ₜʲ(φₙ) = w̃ₜʲ(φₙ) / (1/M) ∑ w̃ₜʲ(φₙ)
"""
function correction!(inc_weights::Vector{Float64}, norm_weights::Vector{Float64},
                     coeff_terms::V, log_e_terms::V, n_obs::Int) where V<:AbstractVector{Float64}

    # Compute incremental weights
    n_particles = length(inc_weights)

    for i = 1:n_particles
        inc_weights[i] = coeff_terms[i] * exp(log_e_terms[i])
    end

    # Normalize weights
    norm_weights .= inc_weights ./ mean(inc_weights)

    return nothing
end