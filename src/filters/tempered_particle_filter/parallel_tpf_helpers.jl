function tpf_helper!(coeff_terms::S, log_e_1_terms::S, log_e_2_terms::S, φ_old::Float64,
                     Ψ_allstates::Function, y_t::Vector{Float64}, s_t_nontemp::DArray{Float64}, det_HH_t::Float64, inv_HH_t::Array{Float64,2},
                     n_obs_t::Int, stage::Int, inc_weights::S, norm_weights::S,
                     s_t1_temp::DArray{Float64}, ϵ_t::DArray{Float64},
                     unnormalized_wts::S,
                     poolmodel::Bool = false, resampling_method::Symbol = :multinomial,
                     verbose::Symbol = :high) where S<:DArray{Float64,1}

    ### 1. Correction
    # Modifies coeff_terms, log_e_1_terms, log_e_2_terms
    weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                   Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t;
                   initialize = stage == 1,
                   poolmodel = poolmodel)

    φ_new = 1.0 ## Function only runs w/ Bootstrap PF so this is 1.0

    if VERBOSITY[verbose] >= VERBOSITY[:high]
        @show φ_new
    end

    # Modifies inc_weights, norm_weights
    correction!(inc_weights, norm_weights, φ_new, coeff_terms,
                log_e_1_terms, log_e_2_terms, n_obs_t)

    ### 2. Selection
    # Modifies s_t1_temp, s_t_nontemp, ϵ_t
    ## Only need to resample s_t_nontemp when no mutation b/c rest reset in next time iteration.
    selection!(norm_weights, s_t_nontemp;
               resampling_method = resampling_method)

    # Mutation never called for BSPF b/c stage == 1 always

    # unnormalized_wts[:L][:] .= unnormalized_wts[:L] .* inc_weights[:L]
    unnormalized_wts[:L][:] .= mean(unnormalized_wts[:L] .* inc_weights[:L])

    return nothing
end

## This function necessary to remove keywords in call for spmd in first iteration
function adaptive_weight_kernel!(coeff_terms::S, log_e_1_terms::S, log_e_2_terms::S, φ_old::Float64,
                                 Ψ_allstates::Function, y_t::Vector{Float64}, s_t_nontemp::DArray{Float64}, det_HH_t::Float64, inv_HH_t::Array{Float64,2},
                                 initialize::Bool = false, poolmodel::Bool = false) where S<:DArray{Float64,1}
    # Modifies coeff_terms, log_e_1_terms, log_e_2_terms
    weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                   Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t;
                   initialize = initialize,
                   poolmodel = poolmodel)
end

function adaptive_tempered_iter!(coeff_terms::S, log_e_1_terms::S, log_e_2_terms::S, φ_old::Float64,
                                 Ψ_allstates::Function, y_t::Vector{Float64}, s_t_nontemp::DArray{Float64}, det_HH_t::Float64, inv_HH_t::Array{Float64,2},
                                 s_t1_temp::DArray{Float64}, ϵ_t::DArray{Float64}, c_vec::S,
                                 Φ::Function, Ψ_t::Function, QQ::Matrix{Float64}, stage::Int, accept_rate::S,
                                 poolmodel::Bool = false, target_accept_rate::AbstractFloat = 0.4, n_mh_steps::Int = 1,
                                 verbose::Symbol = :high) where S<:DArray{Float64,1}
    if stage > 2 ## 2 b/c first call is w/ stage == 2
        ## c updated separately for each worker. This allows
        ## proposal covariance matrix to scale differently by worker
        ## so we match acceptance rates for all workers,
        ## which should be better.
        c_vec[:L][1] = update_c(c_vec[:L][1], accept_rate[:L][1], target_accept_rate)
        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @show c_vec[:L][1]
            println("------------------------------")
        end

        accept_rate[:L][1] = mutation!(Φ, Ψ_t, QQ, det_HH_t, inv_HH_t, φ_old, y_t,
                                s_t_nontemp, s_t1_temp, ϵ_t, c_vec[:L][1], n_mh_steps;
                                poolmodel = poolmodel)
    end

    ### 1. Correction
    # Modifies coeff_terms, log_e_1_terms, log_e_2_terms
    weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                   Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t;
                   initialize = stage == 1,
                   poolmodel = poolmodel)
end

function adaptive_correction!(inc_weights::S, norm_weights::S, φ_new::Float64, coeff_terms::S,
                              log_e_1_terms::S, log_e_2_terms::S, n_obs_t::Int, unnormalized_wts::S,
                              s_t1_temp::DArray{Float64}, s_t_nontemp::DArray{Float64}, ϵ_t::DArray{Float64}, resampling_method::Symbol = :multinomial) where S<:DArray{Float64,1}

    correction!(inc_weights, norm_weights, φ_new, coeff_terms,
                log_e_1_terms, log_e_2_terms, n_obs_t)

    selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t;
               resampling_method = resampling_method)

    unnormalized_wts[:L][:] .= mean(unnormalized_wts[:L] .* inc_weights[:L])
end

function one_iter!(coeff_terms::S, log_e_1_terms::S, log_e_2_terms::S, φ_old::Float64,
                   Ψ_allstates::Function, y_t::Vector{Float64}, s_t_nontemp::DArray{Float64}, det_HH_t::Float64, inv_HH_t::Array{Float64,2},
                   n_obs_t::Int, stage::Int, inc_weights::S, norm_weights::S,
                   s_t1_temp::DArray{Float64}, ϵ_t::DArray{Float64},
                   unnormalized_wts::S,
                   r_star::AbstractFloat = 2.0, poolmodel::Bool = false,
                   fixed_sched::Vector = zeros(0),
                   findroot::Function = bisection, xtol::AbstractFloat = 1e-3,
                   resampling_method::Symbol = :multinomial, n_mh_steps::Int = 1,
                   verbose::Symbol = :high) where S<:DArray{Float64,1}
    ### 1. Correction
    # Modifies coeff_terms, log_e_1_terms, log_e_2_terms
    weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                   Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t;
                   initialize = stage == 1,
                   poolmodel = poolmodel)

    φ_new = next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs_t,
                   r_star, stage; fixed_sched = fixed_sched, findroot = findroot,
                   xtol = xtol)

    if VERBOSITY[verbose] >= VERBOSITY[:high]
        @show φ_new
    end

    # Modifies inc_weights, norm_weights
    correction!(inc_weights, norm_weights, φ_new, coeff_terms,
                log_e_1_terms, log_e_2_terms, n_obs_t)

    selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t;
               resampling_method = resampling_method)

    unnormalized_wts[:L][:] .= mean(unnormalized_wts[:L] .* inc_weights[:L])
end

function tempered_iter!(coeff_terms::S, log_e_1_terms::S, log_e_2_terms::S, φ_old::Float64,
                        Ψ_allstates::Function, y_t::Vector{Float64}, s_t_nontemp::DArray{Float64}, det_HH_t::Float64, inv_HH_t::Array{Float64,2},
                        n_obs_t::Int, stage::Int, inc_weights::S, norm_weights::S,
                        s_t1_temp::DArray{Float64}, ϵ_t::DArray{Float64}, c_vec::S,
                        Φ::Function, Ψ_t::Function, QQ::Matrix{Float64}, unnormalized_wts::S,
                        accept_rate::S,
                        r_star::AbstractFloat = 2.0, poolmodel::Bool = false,
                        fixed_sched::Vector = zeros(0),
                        findroot::Function = bisection, xtol::AbstractFloat = 1e-3,
                        resampling_method::Symbol = :multinomial, target_accept_rate::AbstractFloat = 0.4,
                        n_mh_steps::Int = 1,
                        verbose::Symbol = :high) where S<:DArray{Float64,1}
    if stage > 2
        c_vec[:L][1] = update_c(c_vec[:L][1], accept_rate[:L][1], target_accept_rate)
        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @show c_vec[:L][1]
            println("------------------------------")
        end
        accept_rate[:L][1] = mutation!(Φ, Ψ_t, QQ, det_HH_t, inv_HH_t, φ_old, y_t,
                                s_t_nontemp, s_t1_temp, ϵ_t, c_vec[:L][1], n_mh_steps;
                                poolmodel = poolmodel)

        #=c_vec[:L][1] = update_c(c_vec[:L][1], accept_rate, target_accept_rate)
        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @show c_vec[:L][1]
            println("------------------------------")
        end=#
    end

    ### 1. Correction
    # Modifies coeff_terms, log_e_1_terms, log_e_2_terms
    weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                   Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t;
                   initialize = stage == 1,
                   poolmodel = poolmodel)

    φ_new = next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs_t,
                   r_star, stage; fixed_sched = fixed_sched, findroot = findroot,
                   xtol = xtol)

    if VERBOSITY[verbose] >= VERBOSITY[:high]
        @show φ_new
    end

    # Modifies inc_weights, norm_weights
    correction!(inc_weights, norm_weights, φ_new, coeff_terms,
                log_e_1_terms, log_e_2_terms, n_obs_t)

    selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t;
               resampling_method = resampling_method)

    unnormalized_wts[:L][:] .= mean(unnormalized_wts[:L] .* inc_weights[:L])
end

function tempered_iter_test!(coeff_terms::S, log_e_1_terms::S, log_e_2_terms::S, φ_old::Float64,
                             Ψ_allstates::Function, y_t::Vector{Float64}, s_t_nontemp::DArray{Float64}, det_HH_t::Float64, inv_HH_t::Array{Float64,2},
                             n_obs_t::Int, stage::Int, inc_weights::S, norm_weights::S,
                             s_t1_temp::DArray{Float64}, ϵ_t::DArray{Float64}, c_vec::S,
                             Φ::Function, Ψ_t::Function, QQ::Matrix{Float64},
                             accept_rate::S,
                             r_star::AbstractFloat = 2.0, poolmodel::Bool = false,
                             fixed_sched::Vector = zeros(0),
                             findroot::Function = bisection, xtol::AbstractFloat = 1e-3,
                             target_accept_rate::AbstractFloat = 0.4,
                             n_mh_steps::Int = 1,
                             verbose::Symbol = :high) where S<:DArray{Float64,1}
    if stage > 2
        c_vec[:L][1] = update_c(c_vec[:L][1], accept_rate[:L][1], target_accept_rate)
        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @show c_vec[:L][1]
            println("------------------------------")
        end
        accept_rate[:L][1] = mutation!(Φ, Ψ_t, QQ, det_HH_t, inv_HH_t, φ_old, y_t,
                                s_t_nontemp, s_t1_temp, ϵ_t, c_vec[:L][1], n_mh_steps;
                                poolmodel = poolmodel)

        #=c_vec[:L][1] = update_c(c_vec[:L][1], accept_rate, target_accept_rate)
        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @show c_vec[:L][1]
            println("------------------------------")
        end=#
    end

    ### 1. Correction
    # Modifies coeff_terms, log_e_1_terms, log_e_2_terms
    weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                   Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t;
                   initialize = stage == 1,
                   poolmodel = poolmodel)

    φ_new = next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs_t,
                   r_star, stage; fixed_sched = fixed_sched, findroot = findroot,
                   xtol = xtol)

    if VERBOSITY[verbose] >= VERBOSITY[:high]
        @show φ_new
    end

    # Modifies inc_weights, norm_weights
    correction!(inc_weights, norm_weights, φ_new, coeff_terms,
                log_e_1_terms, log_e_2_terms, n_obs_t)
end


function selection_test!(norm_weights::S, s_t1_temp::DArray{Float64}, s_t_nontemp::DArray{Float64}, ϵ_t::DArray{Float64}, unnormalized_wts::S, inc_weights::S,
               resampling_method = :multinomial) where S<:DArray{Float64,1}

    selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t;
               resampling_method = resampling_method)

    unnormalized_wts[:L][:] .= mean(unnormalized_wts[:L] .* inc_weights[:L])
end
