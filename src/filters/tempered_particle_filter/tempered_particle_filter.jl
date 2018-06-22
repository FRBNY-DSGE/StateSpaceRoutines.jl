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
- `Ψ`: The measurement equation: y_t = Ψ(s_t) + u_t
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
function tempered_particle_filter(data::Matrix{S}, Φ::Function, Ψ::Function,
                                  F_ϵ::Distribution, F_u::Distribution, s_init::Matrix{S};
                                  verbose::Symbol = :high, fixed_sched::Vector{S} = zeros(0),
                                  r_star::S = 2., c::S = 0.3, accept_rate::S = 0.4, target::S = 0.4,
                                  xtol::Float64 = 1e-3, findroot::Function = bisection,
                                  resampling_method = :multinomial,
                                  N_MH::Int = 1, n_particles::Int = 1000,
                                  n_presample_periods::Int = 0,
                                  allout::Bool = true, parallel::Bool = false,
                                  testing::Bool = false) where S<:AbstractFloat
    #--------------------------------------------------------------
    # Setup
    #--------------------------------------------------------------

    adaptive = isempty(fixed_sched)

    # Ensuring the fixed φ schedule is bounded properly
    if !adaptive
        @assert fixed_sched[end] == 1 "φ schedule must be a range from [a,1] s.t. a > 0."
    end

    # Initialization of constants and output vectors
    n_obs, T  = size(data)
    n_shocks  = length(F_ϵ)
    n_states  = size(s_init, 1)
    lik       = zeros(T)
    times     = zeros(T)

    # Array types depend on parallelization
    MyVector = parallel ? SharedVector : Vector
    MyMatrix = parallel ? SharedMatrix : Matrix

    #--------------------------------------------------------------
    # Main Algorithm: Tempered Particle Filter
    #--------------------------------------------------------------

    # Draw initial particles from the distribution of s₀: N(s₀, P₀)
    s_t1_temp = s_init

    # Vectors of the 3 component terms that are used to calculate the weights
    # Inputs saved in these vectors to conserve memory/avoid unnecessary re-computation
    coeff_terms   = MyVector{Float64}(n_particles)
    log_e_1_terms = MyVector{Float64}(n_particles)
    log_e_2_terms = MyVector{Float64}(n_particles)

    for t = 1:T
        tic()
        if VERBOSITY[verbose] >= VERBOSITY[:low]
            println("============================================================")
            @show t
        end

        #--------------------------------------------------------------
        # Initialization
        #--------------------------------------------------------------

        # Remove rows/columns of series with NaN values
        y_t = data[:,t]

        nonmissing  = isfinite.(y_t)
        y_t         = y_t[nonmissing]
        n_obs_t     = length(y_t)
        Ψ_t         = x -> Ψ(x)[nonmissing]
        HH_t        = F_u.Σ.mat[nonmissing, nonmissing]
        inv_HH_t    = inv(HH_t)
        det_HH_t    = det(HH_t)

        # Initialize s_t_nontemp and ϵ by drawing ϵ and forecasting
        ϵ = MyMatrix{Float64}(n_shocks, n_particles)
        s_t_nontemp = MyMatrix{Float64}(n_states, n_particles)

        @mypar parallel for i in 1:n_particles
            ϵ[:, i] = rand(F_ϵ)
            s_t_nontemp[:, i] = Φ(s_t1_temp[:, i], ϵ[:, i])
        end

        # Initialize weight vectors
        inc_weights  = Vector{Float64}(n_particles)
        norm_weights = Vector{Float64}(n_particles)

        # Tempering initialization
        φ_old = 1e-30
        stage = 0

        #--------------------------------------------------------------
        # Main Algorithm
        #--------------------------------------------------------------
        while φ_old < 1
            stage += 1

            # Correction
            φ_new = next_φ!(Ψ_t, stage, φ_old, det_HH_t, inv_HH_t, y_t, s_t_nontemp,
                            coeff_terms, log_e_1_terms, log_e_2_terms, r_star;
                            adaptive = adaptive, findroot = findroot, xtol = xtol,
                            fixed_sched = fixed_sched, parallel = parallel)

            correction!(φ_new, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs_t,
                        inc_weights, norm_weights)

            if VERBOSITY[verbose] >= VERBOSITY[:high]
                @show φ_new
            end

            # Selection
            selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ; resampling_method = resampling_method)

            lik[t] += log(mean(inc_weights))
            c = update_c!(c, accept_rate, target)

            if VERBOSITY[verbose] >= VERBOSITY[:high]
                @show c
                println("------------------------------")
            end

            # Mutate particles
            if stage != 1
                accept_rate = mutation!(Φ, Ψ_t, F_ϵ.Σ.mat, det_HH_t, inv_HH_t, φ_new, y_t,
                                        s_t_nontemp, s_t1_temp, ϵ, c, N_MH; parallel = parallel)
            end

            φ_old = φ_new
        end # of loop over stages

        if VERBOSITY[verbose] >= VERBOSITY[:low]
            print("\n")
            @show lik[t]
            print("Completion of one period ")
            times[t] = toc()
        else
            times[t] = toq()
        end
        s_t1_temp = s_t_nontemp
    end # of loop over periods

    if VERBOSITY[verbose] >= VERBOSITY[:low]
        println("=============================================")
    end

    if allout
        return sum(lik[n_presample_periods + 1:end]), lik[n_presample_periods + 1:end], times
    else
        return sum(lik[n_presample_periods + 1:end])
    end
end
