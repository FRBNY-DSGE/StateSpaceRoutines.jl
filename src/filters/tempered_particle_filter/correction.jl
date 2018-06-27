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
        n_particles  = length(coeff_terms)
        inc_weights  = Vector{Float64}(n_particles)
        norm_weights = Vector{Float64}(n_particles)
        ineff0(φ) =
            ineff!(inc_weights, norm_weights, φ, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs) - r_star

        if stage == 1 || (sign(ineff0(φ_old)) != sign(ineff0(1.0)))
            # Solve for optimal φ if either
            # 1. First stage
            # 2. Sign change from ineff0(φ_old) to ineff0(1.0)
            return findroot(ineff0, φ_old, 1.0, xtol = xtol)
        else
            # Otherwise, set φ = 1
            return 1.0
        end
    else
        return fixed_sched[stage]
    end
end

function correction!(inc_weights::Vector{Float64}, norm_weights::Vector{Float64},
                     φ_new::Float64, coeff_terms::V, log_e_1_terms::V, log_e_2_terms::V,
                     n_obs::Int) where V<:AbstractVector{Float64}
    # Compute incremental weights
    n_particles = length(inc_weights)
    for i = 1:n_particles
        inc_weights[i] =
            φ_new^(n_obs/2) * coeff_terms[i] * exp(log_e_1_terms[i]) * exp(φ_new*log_e_2_terms[i])
    end

    # Normalize weights
    norm_weights .= inc_weights ./ mean(inc_weights)

    return nothing
end

function ineff!(inc_weights::Vector{Float64}, norm_weights::Vector{Float64},
                φ_new::Float64, coeff_terms::V, exp_1_terms::V, exp_2_terms::V,
                n_obs::Int) where V<:AbstractVector{Float64}

    correction!(inc_weights, norm_weights, φ_new, coeff_terms, exp_1_terms, exp_2_terms, n_obs)
    return sum(norm_weights.^2) / length(norm_weights)
end