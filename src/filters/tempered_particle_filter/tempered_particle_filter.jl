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
                         allout::Bool = true, parallel::Bool = false)
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

### Outputs

- `sum(lik)`: The tempered particle filter approximated log-likelihood
- `lik`: (`hist_periods` x 1) vector returning log-likelihood per period t
- `times`: (`hist_periods` x 1) vector returning elapsed runtime per period t

"""
function tempered_particle_filter{S<:AbstractFloat}(data::Matrix{S}, Φ::Function, Ψ::Function,
                                                    F_ϵ::Distribution, F_u::Distribution, s_init::Matrix{S};
                                                    verbose::Symbol = :high, fixed_sched::Vector{S} = zeros(0),
                                                    r_star::S = 2., c::S = 0.3, accept_rate::S = 0.4, target::S = 0.4,
                                                    xtol::S = 0., resampling_method = :systematic, N_MH::Int = 1,
                                                    n_particles::Int = 1000, n_presample_periods::Int = 0,
                                                    adaptive::Bool = true, allout::Bool = true, parallel::Bool = false)
    #--------------------------------------------------------------
    # Setup
    #--------------------------------------------------------------

    # Ensuring the fixed φ schedule is bounded properly
    if !adaptive
        try
            @assert fixed_sched[1] > 0. && fixed_sched[1] < 1.
            @assert fixed_sched[end] == 1.
        catch
            throw("Invalid fixed φ schedule. It must be a range from [a,1] s.t. a > 0.")
        end
    end

    # Initialization of constants and output vectors
    n_observables = size(data, 1)
    n_states      = size(s_init, 1)
    T             = size(data, 2)
    lik           = zeros(T)
    times         = zeros(T)

    # Ensuring Φ, Ψ broadcast to matrices
    Φ_bcast(s_t1::Matrix{S}, ϵ_t1::Matrix{S}) = hcat([Φ(s_t1[:,i], ϵ_t1[:,i]) for i in 1:size(s_t1, 2)]...)
    Ψ_bcast(s_t1::Matrix{S}, u_t1::Matrix{S}) = hcat([Ψ(s_t1[:,i], u_t1[:,i]) for i in 1:size(s_t1, 2)]...)

    #--------------------------------------------------------------
    # Main Algorithm: Tempered Particle Filter
    #--------------------------------------------------------------

    # Draw initial particles from the distribution of s₀: N(s₀, P₀)
    s_lag_tempered = s_init

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
        nonmissing          = !isnan(y_t)
        y_t                 = y_t[nonmissing]
        n_observables_t     = length(y_t)
        Ψ_bcast_t           = (x, ϵ) -> Ψ_bcast(x, ϵ)[nonmissing, :]
        HH_t                = F_u.Σ.mat[nonmissing, nonmissing]

        # Draw random shock ϵ
        ϵ = rand(F_ϵ, n_particles)

        # Forecast forward one time step
        s_t_nontempered = Φ_bcast(s_lag_tempered, ϵ)

        # Error for each particle
        p_error = y_t .- Ψ_bcast_t(s_t_nontempered, zeros(n_observables_t, n_particles))

        # Solve for initial tempering parameter φ_1
        if adaptive
            init_Ineff_func(φ) = solve_inefficiency(φ, 2.0*pi, y_t, p_error, HH_t;
                                                    initialize = true) - r_star
            φ_1 = fzero(init_Ineff_func, 1e-30, 1.0, xtol = xtol)
        else
            φ_1 = fixed_sched[1]
        end

        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @show φ_1
            println("------------------------------")
        end

        # Correct and resample particles
        loglik, id = correction_selection!(φ_1, 0.0, y_t, p_error, HH_t, n_particles,
                                           initialize = true, parallel = parallel,
                                           resampling_method = resampling_method)

        # Update likelihood
        lik[t] += loglik

        # Update arrays for resampled indices
        s_lag_tempered  = s_lag_tempered[:, id]
        s_t_nontempered = s_t_nontempered[:, id]
        ϵ               = ϵ[:, id]

        # Tempering initialization
        φ_old = φ_1
        count = 1

        #--------------------------------------------------------------
        # Main Algorithm
        #--------------------------------------------------------------
        while φ_old < 1

            count += 1

            # Get error for all particles
            p_error = y_t .- Ψ_bcast_t(s_t_nontempered, zeros(n_observables_t, n_particles))

            # Define inefficiency function
            init_ineff_func(φ) = solve_inefficiency(φ, φ_old, y_t, p_error, HH_t) - r_star
            fphi_interval = [init_ineff_func(φ_old) init_ineff_func(1.0)]

            # The below boolean checks that a solution exists within interval
            if prod(sign(fphi_interval)) == -1 && adaptive
                φ_new = fzero(init_ineff_func, φ_old, 1., xtol=xtol)
            elseif prod(sign(fphi_interval)) != -1 && adaptive
                φ_new = 1.
            else # fixed φ
                φ_new = fixed_sched[count]
            end

            if VERBOSITY[verbose] >= VERBOSITY[:high]
                @show φ_new
            end

            # Correct and resample particles
            loglik, id = correction_selection!(φ_new, φ_old, y_t, p_error, HH_t, n_particles;
                                              parallel = parallel)

            # Update likelihood
            lik[t] += loglik

            # Update arrays for resampled indices
            s_lag_tempered  = s_lag_tempered[:,id]
            s_t_nontempered = s_t_nontempered[:,id]
            ϵ               = ϵ[:,id]
            p_error         = p_error[:,id]

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

            if parallel
                if VERBOSITY[verbose] >= VERBOSITY[:high]
                    tic()
                    print("(in parallel) ")
                end
                out = @sync @parallel (hcat) for i = 1:n_particles
                    mutation(Φ, Ψ, F_ϵ, F_u, φ_new, y_t, s_t_nontempered[:,i],
                             s_lag_tempered[:,i], ε[:,i], c, N_MH)
                    end
                for i = 1:n_particles
                    s_t_nontempered[:,i] = out[i][1]
                    ϵ[:,i] = out[i][2]
                    accept_vec[i] = out[i][3]
                end
            else
                if VERBOSITY[verbose] >= VERBOSITY[:high]
                    tic()
                    print("(not parallel) ")
                end
                for i = 1:n_particles
                    s_t_nontempered[:,i], ϵ[:,i], accept_vec[i]  = mutation(Φ, Ψ, F_ϵ, F_u, φ_new, y_t,
                                                                            s_t_nontempered[:,i], s_lag_tempered[:,i],
                                                                            ϵ[:,i], c, N_MH)
                end
            end

            if VERBOSITY[verbose] >= VERBOSITY[:high]
                toc()
            end

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
