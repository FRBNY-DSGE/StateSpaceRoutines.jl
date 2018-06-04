"""
```
tempered_particle_filter{S<:AbstractFloat}(data::Array{S}, Φ::Function,
                         Ψ::Function, F_ϵ::Distribution, F_u::Distribution, s_init::Matrix{S};
                         verbose::Symbol = :high, include_presample::Bool = true,
                         fixed_sched::Vector{S} = zeros(0), r_star::S = 2.,]
                         c::S = 0.3, accept_rate::S = 0.4, target::S = 0.4,
                         xtol = 0., resampling_method::Symbol = :systematic,
                         N_MH::Int = 1, n_particles::Int = 1000,
                         n_presample_periods::Int = 0, adaptive::bool = true,
                         allout::Bool = true, parallel::Bool = false,
                         testing = false)
```
Executes tempered particle filter.

### Inputs

- `data`: (`n_observables` x `hist_periods`) size `Matrix{S}` of data for observables.
- `Φ`: The state transition function: s_t = Φ(s_t-1, ϵ_t)
- `Ψ`: The measurement equation: y_t = Ψ(s_t, u_t)
- `F_ϵ`: The shock distribution: ϵ ~ F_ϵ
- `F_u`: The measurement error distribution: u ~ F_u
- `s_init`: (`n_observables` x `n_particles`) initial state vector

### Keyword Arguments

- `verbose`: Indicates desired nuance of outputs. Default to `:low`.
- `include_presample`: Indicates whether to include presample in periods in the returned
   outputs. Defaults to `true`.
- `fixed_sched`: An array of elements in (0,1] that are monotonically increasing, which
specify the tempering schedule.
- `r_star`: The target ratio such that the chosen φ* satisfies r_star = InEff(φ*) = Sample mean with respect
to the number of particles of the squared, normalized particle weights, W_t^{j,n}(φ_n)^2.
- `c`: The adaptively chosen step size of each proposed move in the mutation step of the tempering iterations portion of
the algorithm.
- `accept_rate`: The rate of the number of particles accepted in the mutation step at each time step, which factors
into the calculation of the adaptively chosen c step.
- `target`: The target acceptance rate, which factors into the calculation of the adaptively chosen c step.
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

### Outputs

- `sum(lik)`: The tempered particle filter approximated log-likelihood
- `lik`: (`hist_periods` x 1) vector returning log-likelihood per period t
- `times`: (`hist_periods` x 1) vector returning elapsed runtime per period t

"""
function tempered_particle_filter{S<:AbstractFloat}(data::Matrix{S}, Φ::Function, Ψ::Function,
                                                    F_ϵ::Distribution, F_u::Distribution, s_init::Matrix{S};
                                                    verbose::Symbol = :high, fixed_sched::Vector{S} = zeros(0),
                                                    r_star::S = 2., c::S = 0.3, accept_rate::S = 0.4, target::S = 0.4,
                                                    tol::Float64 = 1e-3, resampling_method = :multinomial,
                                                    N_MH::Int = 1, n_particles::Int = 1000,
                                                    n_presample_periods::Int = 0,
                                                    allout::Bool = true, parallel::Bool = false,
                                                    testing::Bool = false)
    #--------------------------------------------------------------
    # Setup
    #--------------------------------------------------------------

    adaptive = isempty(fixed_sched)

    # Ensuring the fixed φ schedule is bounded properly
    if !adaptive
        try
            @assert fixed_sched[end] == 1.
        catch
            throw("Invalid fixed φ schedule. It must be a range from [a,1] s.t. a > 0.")
        end
    end

    # Initialization of constants and output vectors
    n_obs, T  = size(data)
    n_shocks  = length(F_ϵ)
    n_states  = size(s_init, 1)
    lik       = zeros(T)
    times     = zeros(T)

    # Ensuring Φ, Ψ broadcast to matrices
    function Φ_bcast(s_t1::Matrix{S}, ϵ_t1::Matrix{S})
        s_t = similar(s_t1)
        for i in 1:n_particles
            s_t[:, i] = Φ(s_t1[:, i], ϵ_t1[:, i])
        end
        return s_t
    end

    function Ψ_bcast(s_t::Matrix{S}, u_t::Matrix{S})
        y_t = similar(u_t)
        for i in 1:n_particles
            y_t[:, i] = Ψ(s_t[:, i], u_t[:, i])
        end
        return y_t
    end

    #--------------------------------------------------------------
    # Main Algorithm: Tempered Particle Filter
    #--------------------------------------------------------------

    # Draw initial particles from the distribution of s₀: N(s₀, P₀)
    s_lag_tempered = s_init

    # Vectors of the 3 component terms that are used to calculate the weights
    # Inputs saved in these vectors to conserve memory/avoid unnecessary re-computation
    coeff_terms = Vector{Float64}(n_particles)
    log_e_1_terms = Vector{Float64}(n_particles)
    log_e_2_terms = Vector{Float64}(n_particles)

    for t = 1:T

        tic()
        if VERBOSITY[verbose] >= VERBOSITY[:low]
            println("============================================================")
            @show t
        end

        #--------------------------------------------------------------
        # Initialize Algorithm: First Tempering Step
        #--------------------------------------------------------------
        y_t = data[:,t]

        # Remove rows/columns of series with NaN values
        nonmissing          = isfinite.(y_t)
        y_t                 = y_t[nonmissing]
        n_obs_t             = length(y_t)
        Ψ_t                 = (x, ϵ) -> Ψ(x, ϵ)[nonmissing]
        Ψ_bcast_t           = (x, ϵ) -> Ψ_bcast(x, ϵ)[nonmissing, :]
        HH_t                = F_u.Σ.mat[nonmissing, nonmissing]
        inv_HH_t            = inv(HH_t)
        det_HH_t            = det(HH_t)

        #####################################
        if parallel
            ϵ = Matrix{Float64}(n_shocks, n_particles)
            s_t_nontempered = similar(s_lag_tempered)
            ϵ, s_t_nontempered, coeff_terms, log_e_1_terms, log_e_2_terms =
            @parallel (vector_reduce) for i in 1:n_particles
                ε = rand(F_ϵ)
                s_t_non = Φ(s_lag_tempered[:, i], ε)
                p_err   = y_t - Ψ_t(s_t_non, zeros(n_obs_t))
                coeff_term, log_e_1_term, log_e_2_term = weight_kernel(0., y_t, p_err, det_HH_t, inv_HH_t,
                                                                       initialize = true)
                vector_reshape(ε, s_t_non, coeff_term, log_e_1_term, log_e_2_term)
            end
            coeff_terms = squeeze(coeff_terms, 1)
            log_e_1_terms = squeeze(log_e_1_terms, 1)
            log_e_2_terms = squeeze(log_e_2_terms, 1)
        else
            # Draw random shock ϵ
            ϵ = rand(F_ϵ, n_particles)

            # Forecast forward one time step
            s_t_nontempered = Φ_bcast(s_lag_tempered, ϵ)

            # Error for each particle
            p_error = y_t .- Ψ_bcast_t(s_t_nontempered, zeros(n_obs_t, n_particles))

            # Solve for initial tempering parameter φ_1
            for i in 1:n_particles
                coeff_terms[i], log_e_1_terms[i], log_e_2_terms[i] = weight_kernel(0., y_t, p_error[:, i],
                                                                                   det_HH_t, inv_HH_t,
                                                                                   initialize = true)
            end
        end

        if adaptive
            init_Ineff_func(φ) = solve_inefficiency(φ, coeff_terms, log_e_1_terms,
                                                    log_e_2_terms, n_obs_t,
                                                    parallel = false) - r_star

            φ_1 = bisection(init_Ineff_func, 1e-30, 1.0, tol = tol)
            # φ_1 = fzero(init_Ineff_func, 1e-30, 1., xtol = 0.)
        else
            φ_1 = fixed_sched[1]
        end
        # #####################################

        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @show φ_1
            println("------------------------------")
        end

        normalized_weights, loglik = correction(φ_1, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs_t)
        s_lag_tempered, s_t_nontempered, ϵ = selection(normalized_weights, s_lag_tempered,
                                                       s_t_nontempered, ϵ; resampling_method = resampling_method)

        # Update likelihood
        lik[t] += loglik

        # Tempering initialization
        φ_old = φ_1
        count = 1

        #--------------------------------------------------------------
        # Main Algorithm
        #--------------------------------------------------------------
        while φ_old < 1

            count += 1

            # Get error for all particles
            p_error = y_t .- Ψ_bcast_t(s_t_nontempered, zeros(n_obs_t, n_particles))

            for i in 1:n_particles
                coeff_terms[i], log_e_1_terms[i], log_e_2_terms[i] = weight_kernel(φ_old, y_t, p_error[:, i],
                                                                                   det_HH_t, inv_HH_t,
                                                                                   initialize = false)
            end

            # Define inefficiency function
            init_ineff_func(φ) = solve_inefficiency(φ, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs_t,
                                                    parallel = false) - r_star
            fphi_interval = [init_ineff_func(φ_old) init_ineff_func(1.0)]

            # The below boolean checks that a solution exists within interval
            if prod(sign.(fphi_interval)) == -1 && adaptive
                φ_new = bisection(init_ineff_func, φ_old, 1., tol = tol)
                # φ_new = fzero(init_ineff_func, φ_old, 1., xtol = 0.)
            elseif prod(sign.(fphi_interval)) != -1 && adaptive
                φ_new = 1.
            else # fixed φ
                φ_new = fixed_sched[count]
            end

            if VERBOSITY[verbose] >= VERBOSITY[:high]
                @show φ_new
            end

            # Correct and resample particles
            normalized_weights, loglik = correction(φ_new, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs_t)
            s_lag_tempered, s_t_nontempered, ϵ = selection(normalized_weights, s_lag_tempered,
                                                           s_t_nontempered, ϵ; resampling_method = resampling_method)

            # Update likelihood
            lik[t] += loglik

            # Update value for c
            c = update_c!(c, accept_rate, target)

            if VERBOSITY[verbose] >= VERBOSITY[:high]
                @show c
                println("------------------------------")
            end

            # Mutation Step
            accept_vec = zeros(n_particles)
            if VERBOSITY[verbose] >= VERBOSITY[:high]
                print("Mutation ")
            end

            s_t_nontempered, ϵ, accept_rate = mutation(Φ, Ψ_t, F_ϵ.Σ.mat, det_HH_t, inv_HH_t, φ_new, y_t,
                                                       s_t_nontempered, s_lag_tempered, ϵ, c, N_MH;
                                                       parallel = parallel)

            # if VERBOSITY[verbose] >= VERBOSITY[:high]
                # toc()
            # end

            # Calculate average acceptance rate
            accept_rate = mean(accept_vec)

            # Update φ
            φ_old = φ_new
        end

    if VERBOSITY[verbose] >= VERBOSITY[:high]
        println("Out of main while-loop.")
    end

    if VERBOSITY[verbose] >= VERBOSITY[:low]
        print("\n")
        @show lik[t]
        print("Completion of one period ")
        times[t] = toc()
    else
        times[t] = toq()
    end
    s_lag_tempered = s_t_nontempered
    end

    if VERBOSITY[verbose] >= VERBOSITY[:low]
        println("=============================================")
    end

    if allout
        return sum(lik[n_presample_periods + 1:end]), lik[n_presample_periods + 1:end], times
    else
        return sum(lik[n_presample_periods + 1:end])
    end
end
