function get_local_inds(x, replaced)
    x[:L][1:replaced]
end

function set_dvals2(x, replaced, proc_j)
    tmp = copy(x[:L][1:replaced])
    x[:L][1:replaced] = remotecall_fetch(get_local_inds, proc_j, x, replaced)
    passobj(proc_i, proc_j, :tmp)
end

function set_dvals2_mat(x, replaced, proc_j)
    tmp = copy(x[:L][:,1:replaced])
    x[:L][:,1:replaced] = remotecall_fetch(get_local_inds, proc_j, x, replaced)
    passobj(proc_i, proc_j, :tmp)
end

function set_dvals3(x, replaced, tmps)
    x[:L][1:replaced] = tmps
end

function set_dvals3_mat(x, replaced, tmps)
    x[:L][:,1:replaced] = tmps
end

function set_dvals4(x, replaced, proc_j)
    tmp1 = copy(x[:L][1:replaced])
    x[:L][1:replaced] = remotecall_fetch(get_local_inds, proc_j, x, replaced)
    passobj(proc_i, proc_j, :tmp1)
end

function set_dvals4_mat(x, replaced, proc_j)
    tmp1 = copy(x[:L][:,1:replaced])
    x[:L][:,1:replaced] = remotecall_fetch(get_local_inds, proc_j, x, replaced)
    passobj(proc_i, proc_j, :tmp1)
end

function tpf_helper!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                     Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                     n_obs_t, stage, inc_weights, norm_weights,
                     s_t1_temp, ϵ_t,
                     Φ, Ψ_t, QQ, unnormalized_wts,
                     poolmodel::Bool = false,
                     fixed_sched::Vector{S} = zeros(0),
                     findroot::Function = bisection, xtol::S = 1e-3,
                     resampling_method::Symbol = :multinomial, target_accept_rate::S = 0.4,
                     accept_rate::S = target_accept_rate, n_mh_steps::Int = 1,
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
                                 Φ, Ψ_t, QQ, stage,
                                 poolmodel::Bool = false, target_accept_rate::S = 0.4,
                                 accept_rate::S = target_accept_rate, n_mh_steps::Int = 1,
                                 verbose::Symbol = :high) where S<:AbstractFloat
    if stage > 2 ## 2 b/c first call is w/ stage == 2
        accept_rate = mutation!(Φ, Ψ_t, QQ, det_HH_t, inv_HH_t, φ_old, y_t,
                                s_t_nontemp, s_t1_temp, ϵ_t, c_vec[:L][1], n_mh_steps;
                                poolmodel = poolmodel)

        ## c updated separately for each worker. This allows
        ## proposal covariance matrix to scale differently by worker
        ## so we match acceptance rates for all workers,
        ## which should be better.

        # No need to update if no mutation step
        c_vec[:L][1] = update_c(c_vec[:L][1], accept_rate, target_accept_rate)
        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @show c
            println("------------------------------")
        end
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
                   resampling_method::Symbol = :multinomial, target_accept_rate::S = 0.4,
                   accept_rate::S = target_accept_rate, n_mh_steps::Int = 1,
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
                        accept_rate::S = target_accept_rate, n_mh_steps::Int = 1,
                        verbose::Symbol = :high) where S<:AbstractFloat
    if stage > 2
        accept_rate = mutation!(Φ, Ψ_t, QQ, det_HH_t, inv_HH_t, φ_old, y_t,
                                s_t_nontemp, s_t1_temp, ϵ_t, c_vec[:L][1], n_mh_steps;
                                poolmodel = poolmodel)

        c_vec[:L][1] = update_c(c_vec[:L][1], accept_rate, target_accept_rate)
        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @show c_vec[:L][1]
            println("------------------------------")
        end
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
