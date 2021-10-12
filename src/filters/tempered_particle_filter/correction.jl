"""
```
weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψ, y_t,
    s_t_nontemp, det_HH, inv_HH; initialize = false, parallel = false,
    poolmodel = false)
```

The outputs of the weight_kernel function are meant to speed up the adaptive φ
finding, so that we don't do the same matrix multiplication step in every
iteration of the root-solving algorithm.

The exponential terms are logged first and then exponentiated in the
incremental weight calculation so the problem is well-conditioned (i.e. not
exponentiating very large negative numbers).

This function modifies `coeff_terms`, `log_e_1_terms`, and `log_e_2_terms`.
"""
function weight_kernel!(coeff_terms::V, log_e_1_terms::V, log_e_2_terms::V,
                        φ_old::Float64, Ψ::Function, y_t::Vector{Float64},
                        s_t_nontemp::AbstractMatrix{Float64},
                        det_HH::Float64, inv_HH::Matrix{Float64};
                        initialize::Bool = false,
                        parallel::Bool = false,
                        poolmodel::Bool = false) where V<:AbstractVector{Float64}
    # Sizes
    n_particles = parallel ? length(coeff_terms[:L]) : length(coeff_terms)
    n_obs = length(y_t)

    if poolmodel
        if parallel
            if initialize
                for i in 1:n_particles
                    # Initialization step (using 2π instead of φ_old)
                    coeff_terms[:L][i] = (2*pi)^(-n_obs/2) # this may need to be adjusted
                    log_e_1_terms[:L][i] = 0.
                    log_e_2_terms[:L][i] = log(Ψ(s_t_nontemp[:,i]))
                end
            else
                for i in 1:n_particles
                    coeff_terms[:L][i] = (φ_old)^(-n_obs/2)
                    log_e_1_terms[:L][i] = -φ_old * log(Ψ(s_t_nontemp[:,i]))
                    log_e_2_terms[:L][i] = log(Ψ(s_t_nontemp[:,i]))
                end
            end

#=            if initialize
                coeff_terms[:L] = fill((2*pi)^(-n_obs/2), n_particles)
                log_e_1_terms[:L] = zeros(n_particles)



                for i in 1:n_particles
                    log_e_2_terms[:L][i] = #DArray(size(log_e_2_terms[:L])) do inds
                    arr = zeros(inds)
                    s_t_no = OffsetArray(localpart(s_t_nontemp), DistributedArrays.localindices(s_t_nontemp))
                    for i in inds[1] ## [1] b/c inds is (1:n,) so need to index into tuple first
                        arr[i] = log(Ψ(convert(Vector,s_t_no[:,i])))
                    end
                    parent(arr)
                end
            else
                coeff_terms .= dfill((φ_old)^(-n_obs/2), size(coeff_terms))
                half_wkr = nworkers()#Int(floor(n_workers()/2)) ## TODO: Might not work if each worker has some part of s_t_nontemp and you only use half the workers here.

                log_e_1_terms .= DArray(size(log_e_1_terms), workers()[1:half_wkr], [1,half_wkr]) do inds
                    arr = zeros(inds)
                    s_t_no = OffsetArray(localpart(s_t_nontemp), DistributedArrays.localindices(s_t_nontemp))
                    for i in inds[1]
                        arr[i] = -φ_old * log(Ψ(convert(Vector,s_t_no[:,i])))
                    end
                    parent(arr)
                end

                log_e_2_terms .= DArray(size(log_e_2_terms), workers()[1:half_wkr], [1,half_wkr]) do inds
                    arr = zeros(inds)
                    s_t_no = OffsetArray(localpart(s_t_nontemp), DistributedArrays.localindices(s_t_nontemp))
                    for i in inds[1]
                        arr[i] = log(Ψ(convert(Vector,s_t_no[:,i])))
                    end
                    parent(arr)
                end
            end=#
        else
            for i in 1:n_particles
                if initialize
                    # Initialization step (using 2π instead of φ_old)
                    coeff_terms[i] = (2*pi)^(-n_obs/2) # this may need to be adjusted
                    log_e_1_terms[i] = 0.
                    log_e_2_terms[i] = log(Ψ(s_t_nontemp[:,i]))
                else
                    coeff_terms[i] = (φ_old)^(-n_obs/2)
                    log_e_1_terms[i] = -φ_old * log(Ψ(s_t_nontemp[:,i]))
                    log_e_2_terms[i] = log(Ψ(s_t_nontemp[:,i]))
                end
            end
        end
    else
        if parallel
            if initialize
                for i in 1:n_particles
                    error    = y_t - Ψ(s_t_nontemp[:, i])
                    sq_error = dot(error, inv_HH * error)

                    # Initialization step (using 2π instead of φ_old)
                    coeff_terms[:L][i]   = (2*pi)^(-n_obs/2) * det_HH^(-1/2)
                    log_e_1_terms[:L][i] = 0.
                    log_e_2_terms[:L][i] = -1/2 * sq_error
                end
            else
                for i in 1:n_particles
                    error    = y_t - Ψ(s_t_nontemp[:, i])
                    sq_error = dot(error, inv_HH * error)

                    # Non-initialization step (tempering and final iteration)
                    coeff_terms[:L][i]   = (φ_old)^(-n_obs/2)
                    log_e_1_terms[:L][i] = -1/2 * (-φ_old) * sq_error
                    log_e_2_terms[:L][i] = -1/2 * sq_error
                end
            end
#=
            sq_error = DArray((n_particles,), workers()) do inds
                arr = zeros(inds)
                s_t_no = OffsetArray(localpart(s_t_nontemp), DistributedArrays.localindices(s_t_nontemp))
                for i in inds[1] ## [1] b/c inds is (1:n,) so need to index into tuple first
                    errors = y_t .- Ψ(convert(Vector,s_t_no[:,i]))
                    arr[i] = dot(errors, inv_HH * errors)
                end
                parent(arr)
            end

            sq_error = convert(Array, sq_error)

            if initialize
                # Initialization step (using 2π instead of φ_old)
                coeff_terms   .= dfill((2*pi)^(-n_obs/2) * det_HH^(-1/2), size(coeff_terms))
                log_e_1_terms .= dzeros(size(log_e_1_terms))
                log_e_2_terms .= -1/2 * sq_error#dfill(-1/2 * sq_error, size(log_e_2_terms))
            else
                # Non-initialization step (tempering and final iteration)
                coeff_terms   .= dfill((φ_old)^(-n_obs/2), size(coeff_terms))
                log_e_1_terms .= -1/2 * (-φ_old) * sq_error#dfill(-1/2 * (-φ_old) * sq_error, size(log_e_1_terms))
                log_e_2_terms .= -1/2 * sq_error#dfill(-1/2 * sq_error, size(log_e_2_terms))
            end=#
        else
            for i in 1:n_particles
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
        end
end
return nothing
end

# Parallel version
function weight_kernel!(coeff_terms::V, log_e_1_terms::V, log_e_2_terms::V,
                        φ_old::Float64, Ψ::Function, y_t::Vector{Float64},
                        s_t_nontemp::DArray{Float64,2},
                        det_HH::Float64, inv_HH::Matrix{Float64};
                        initialize::Bool = false,
                        parallel::Bool = true,
                        poolmodel::Bool = false) where V<:DArray{Float64,1}
    # Sizes
    n_particles = length(coeff_terms[:L])
    n_obs = length(y_t)

    if poolmodel
        if parallel
            if initialize
                for i in 1:n_particles
                    # Initialization step (using 2π instead of φ_old)
                    coeff_terms[:L][i] = (2*pi)^(-n_obs/2) # this may need to be adjusted
                    log_e_1_terms[:L][i] = 0.
                    log_e_2_terms[:L][i] = log(Ψ(s_t_nontemp[:,i]))
                end
            else
                for i in 1:n_particles
                    coeff_terms[:L][i] = (φ_old)^(-n_obs/2)
                    log_e_1_terms[:L][i] = -φ_old * log(Ψ(s_t_nontemp[:,i]))
                    log_e_2_terms[:L][i] = log(Ψ(s_t_nontemp[:,i]))
                end
            end
        else
            @show "Why are you running DArray but not in parallel?"
            @assert false
        end
    else
        if parallel
            if initialize
                for i in 1:n_particles
                    error    = y_t - Ψ(s_t_nontemp[:L][:, i])
                    sq_error = dot(error, inv_HH * error)

                    # Initialization step (using 2π instead of φ_old)
                    coeff_terms[:L][i]   = (2*pi)^(-n_obs/2) * det_HH^(-1/2)
                    log_e_1_terms[:L][i] = 0.
                    log_e_2_terms[:L][i] = -1/2 * sq_error
                end
            else
                for i in 1:n_particles
                    error    = y_t - Ψ(s_t_nontemp[:L][:, i])
                    sq_error = dot(error, inv_HH * error)

                    # Non-initialization step (tempering and final iteration)
                    coeff_terms[:L][i]   = (φ_old)^(-n_obs/2)
                    log_e_1_terms[:L][i] = -1/2 * (-φ_old) * sq_error
                    log_e_2_terms[:L][i] = -1/2 * sq_error
                end
            end
        else
            @show "Why are you running DArray but not in parallel?"
            @assert false
        end
    end
    return nothing
end

"""
```
next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs, r_star, stage;
    fixed_sched = [], findroot = bisection, xtol = 1e-3)
```

Return the next tempering factor `φ_new`. If `isempty(fixed_sched)`, adaptively
choose `φ_new` by setting InEff(φ_new) = r_star. Otherwise, return
`fixed_sched[stage]`.
"""
function next_φ(φ_old::Float64, coeff_terms::V, log_e_1_terms::V, log_e_2_terms::V,
                n_obs::Int, r_star::Float64, stage::Int;
                fixed_sched::Vector{Float64} = Float64[],
                findroot::Function = bisection,
                xtol::Float64 = 1e-3, parallel::Bool = false) where V<:AbstractVector{Float64}

    if isempty(fixed_sched)
        n_particles  = parallel ? length(coeff_terms[:L]) : length(coeff_terms)
        inc_weights  = Vector{Float64}(undef, n_particles)
        norm_weights = Vector{Float64}(undef, n_particles)
        if parallel
            ineff0(φ) =
                ineff!(inc_weights[:L], norm_weights[:L], φ, coeff_terms[:L], log_e_1_terms[:L], log_e_2_terms[:L], n_obs) - r_star
        else
            ineff0(φ) =
                ineff!(inc_weights, norm_weights, φ, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs) - r_star
        end

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

## Parallel Version
function next_φ(φ_old::Float64, coeff_terms::V, log_e_1_terms::V, log_e_2_terms::V,
                n_obs::Int, r_star::Float64, stage::Int;
                fixed_sched::Vector{Float64} = Float64[],
                findroot::Function = bisection,
                xtol::Float64 = 1e-3) where V<:DArray{Float64,1}

    if isempty(fixed_sched)
        n_particles  = length(coeff_terms[:L])
        inc_weights  = Vector{Float64}(undef, n_particles)
        norm_weights = Vector{Float64}(undef, n_particles)

        ineff0(φ) =
            ineff!(inc_weights, norm_weights, φ, coeff_terms[:L], log_e_1_terms[:L], log_e_2_terms[:L], n_obs) - r_star

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

"""
```
correction!(inc_weights, norm_weights, φ_new, coeff_terms, log_e_1_terms,
     log_e_2_terms, n_obs)
```

Compute (and modify in-place) incremental weights w̃ₜʲ and normalized weights W̃ₜʲ:

        w̃ₜʲ(φₙ) = pₙ(yₜ|sₜʲ'ⁿ⁻¹) / pₙ₋₁(yₜ|sₜʲ'ⁿ⁻¹)
                = (φₙ/φₙ₋₁)^(d/2) exp{-1/2 (φₙ-φₙ₋₁) [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]' Σᵤ⁻¹ [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]}

        W̃ₜʲ(φₙ) = w̃ₜʲ(φₙ) / (1/M) ∑ w̃ₜʲ(φₙ)
"""
function correction!(inc_weights::Vector{Float64}, norm_weights::Vector{Float64},
                     φ_new::Float64, coeff_terms::V, log_e_1_terms::V, log_e_2_terms::V,
                     n_obs::Int; parallel::Bool = false) where V<:AbstractVector{Float64}
    # Compute incremental weights
    n_particles = parallel ? length(inc_weights[:L]) : length(inc_weights)
    if parallel
        @sync @distributed for i in 1:n_particles
            inc_weights[i] =
                φ_new^(n_obs/2) * coeff_terms[i] * exp(log_e_1_terms[i]) * exp(φ_new*log_e_2_terms[i])
        end
    else
        for i = 1:n_particles
            inc_weights[i] =
                φ_new^(n_obs/2) * coeff_terms[i] * exp(log_e_1_terms[i]) * exp(φ_new*log_e_2_terms[i])
        end
    end

    # Normalize weights
    norm_weights .= inc_weights ./ mean(inc_weights)

    return nothing
end

# Parallel BSPF Version when correction called as part of ineff!
function correction!(inc_weights::Vector{Float64}, norm_weights::Vector{Float64},
                     φ_new::Float64, coeff_terms::V, log_e_1_terms::V, log_e_2_terms::V,
                     n_obs::Int) where V<:DArray{Float64,1}
    # Compute incremental weights
    n_particles = length(inc_weights)
    for i = 1:n_particles
        inc_weights[i] =
            φ_new^(n_obs/2) * coeff_terms[:L][i] * exp(log_e_1_terms[:L][i]) * exp(φ_new*log_e_2_terms[:L][i])
    end

    # Normalize weights
    norm_weights .= inc_weights ./ mean(inc_weights)

    return nothing
end

# Parallel BSPF Version when correction called on its own
function correction!(inc_weights::V, norm_weights::V,
                     φ_new::Float64, coeff_terms::V, log_e_1_terms::V, log_e_2_terms::V,
                     n_obs::Int) where V<:DArray{Float64,1}
    # Compute incremental weights
    n_particles = length(inc_weights[:L])
    for i = 1:n_particles
        inc_weights[:L][i] =
            φ_new^(n_obs/2) * coeff_terms[:L][i] * exp(log_e_1_terms[:L][i]) * exp(φ_new*log_e_2_terms[:L][i])
    end

    # Normalize weights
    norm_weights[:L] = inc_weights[:L] ./ mean(inc_weights[:L])

    return nothing
end

"""
```
ineff!(inc_weights, norm_weights, φ_new, coeff_terms, exp_1_terms, exp_2_terms, n_obs)
```

Compute and return InEff(φₙ), where:

        InEff(φₙ) = (1/M) ∑ (W̃ₜʲ(φₙ))²
"""
function ineff!(inc_weights::Vector{Float64}, norm_weights::Vector{Float64},
                φ_new::Float64, coeff_terms::V, exp_1_terms::V, exp_2_terms::V,
                n_obs::Int) where V<:AbstractVector{Float64}

    correction!(inc_weights, norm_weights, φ_new, coeff_terms, exp_1_terms,
                exp_2_terms, n_obs)
    return sum(norm_weights.^2) / length(norm_weights)
end

# Parallel BSPF Version
function ineff!(inc_weights::V, norm_weights::V,
                φ_new::Float64, coeff_terms::V, exp_1_terms::V, exp_2_terms::V,
                n_obs::Int) where V<:DArray{Float64,1}

    correction!(inc_weights, norm_weights, φ_new, coeff_terms, exp_1_terms,
                exp_2_terms, n_obs)
    return sum(norm_weights.^2) / length(norm_weights)
end
