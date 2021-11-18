function tpf_helper!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                     Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                     n_obs_t, stage, inc_weights, norm_weights,
                     s_t1_temp, ϵ_t,
                     Φ, Ψ_t, QQ, unnormalized_wts,
                     poolmodel::Bool = false,
                     fixed_sched::Vector{S} = zeros(0),
                     findroot::Function = bisection, xtol::S = 1e-3,
                     resampling_method::Symbol = :multinomial, n_mh_steps::Int = 1,
                     verbose::Symbol = :high) where S<:AbstractFloat

    ### 1. Correction
    # Modifies coeff_terms, log_e_1_terms, log_e_2_terms
    weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                   Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t;
                   initialize = stage == 1,
                   poolmodel = poolmodel)

    φ_new = fixed_sched[stage] ## Function only runs w/ Bootstrap PF so this is 1.0

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

## This function to remove keywords in call for spmd in first iteration
function adaptive_weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                                 Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                                 initialize, poolmodel)
    # Modifies coeff_terms, log_e_1_terms, log_e_2_terms
    weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                   Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t;
                   initialize = initialize,
                   poolmodel = poolmodel)
end

function adaptive_tempered_iter!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                                 Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                                 s_t1_temp, ϵ_t, c_vec,
                                 Φ, Ψ_t, QQ, stage, accept_rate,
                                 poolmodel::Bool = false, target_accept_rate::S = 0.4,
                                 n_mh_steps::Int = 1, ##accept_rate is DVector
                                 verbose::Symbol = :high) where S<:AbstractFloat
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

function adaptive_correction!(inc_weights, norm_weights, φ_new, coeff_terms,
                              log_e_1_terms, log_e_2_terms, n_obs_t, unnormalized_wts, s_t1_temp, s_t_nontemp, ϵ_t, resampling_method) where S<:AbstractFloat

    correction!(inc_weights, norm_weights, φ_new, coeff_terms,
                log_e_1_terms, log_e_2_terms, n_obs_t)

    selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t;
               resampling_method = resampling_method)

    unnormalized_wts[:L][:] .= mean(unnormalized_wts[:L] .* inc_weights[:L])
end

function one_iter!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                   Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                   n_obs_t, stage, inc_weights, norm_weights,
                   s_t1_temp, ϵ_t, unnormalized_wts,
                   r_star::S = 2.0, poolmodel::Bool = false,
                   fixed_sched::Vector{S} = zeros(0),
                   findroot::Function = bisection, xtol::S = 1e-3,
                   resampling_method::Symbol = :multinomial, n_mh_steps::Int = 1,
                   verbose::Symbol = :high) where S<:AbstractFloat
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

function tempered_iter!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                        Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                        n_obs_t, stage, inc_weights, norm_weights,
                        s_t1_temp, ϵ_t, c_vec,
                        Φ, Ψ_t, QQ, unnormalized_wts,
                        r_star::S = 2.0, poolmodel::Bool = false,
                        fixed_sched::Vector{S} = zeros(0),
                        findroot::Function = bisection, xtol::S = 1e-3,
                        resampling_method::Symbol = :multinomial, target_accept_rate::S = 0.4,
                        accept_rate = target_accept_rate, n_mh_steps::Int = 1,
                        verbose::Symbol = :high) where S<:AbstractFloat
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

function tempered_iter_test!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                        Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                        n_obs_t, stage, inc_weights, norm_weights,
                        s_t1_temp, ϵ_t, c_vec,
                        Φ, Ψ_t, QQ, unnormalized_wts,
                        r_star::S = 2.0, poolmodel::Bool = false,
                        fixed_sched::Vector{S} = zeros(0),
                        findroot::Function = bisection, xtol::S = 1e-3,
                        resampling_method::Symbol = :multinomial, target_accept_rate::S = 0.4,
                        accept_rate = target_accept_rate, n_mh_steps::Int = 1,
                        verbose::Symbol = :high) where S<:AbstractFloat
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


function selection_test!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t, unnormalized_wts, inc_weights,
               resampling_method)

    selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t;
               resampling_method = resampling_method)

    unnormalized_wts[:L][:] .= mean(unnormalized_wts[:L] .* inc_weights[:L])
end
