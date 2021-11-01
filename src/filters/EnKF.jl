"""
```
tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; verbose = :high,
    n_particles = 1000, fixed_sched = [], r_star = 2, findroot = bisection,
    xtol = 1e-3, resampling_method = :multionial, n_mh_steps = 1, c_init = 0.3,
    target_accept_rate = 0.4, n_presample_periods = 0, allout = true,
    parallel = false, verbose = :low,
    dynamic_measurement = false, poolmodel = false)
```

### Inputs

- `data::Matrix{S}`: `Ny` x `Nt` matrix of historical data
- `Φ::Function`: state transition equation: s_t = Φ(s_{t-1}, ϵ_t)
- `Ψ::Function`: measurement equation: y_t = Ψ(s_t) + u_t
- `F_ϵ::Distribution`: shock distribution: ϵ_t ~ F_ϵ
- `F_u::Distribution`: measurement error distribution: u_t ~ F_u
- `s_init::Matrix{S}`: `Ns` x `n_particles` matrix of initial state vectors s_0

where `S<:AbstractFloat` and

- `Ns` is the number of states
- `Ny` is the number of observables
- `Nt` is the number of data periods

### Keyword Arguments

- `n_particles::Int`: number of particles used to approximate the
  log-likelihood. Using more particles yields a more accurate estimate, at the
  cost of being more computationally intensive

**Correction:**

- `fixed_sched::Vector{S}`: array of monotone increasing values in (0,1] which
  specify the fixed tempering schedule. Set to `[]` if you wish to adaptively
  choose the tempering schedule (below)
- `r_star::S`: target inefficiency ratio used to adaptively choose φ, i.e.
  Ineff(φ_n) := 1/M Σ_{j=1}^M (Wtil_t^{j,n}(φ_n))^2 = r_star
- `findroot::Function`: root-finding algorithm, one of `bisection` or `fzero`
- `xtol::S`: root-finding tolerance on x bracket size

**Selection:**

- `resampling_method::Symbol`: either `:multinomial` or `:systematic`

**Mutation:**

- `n_mh_steps::Int`: number of Metropolis-Hastings steps to take in each
  tempering stage
- `c_init::S`: initial scaling of proposal covariance matrix
- `target_accept_rate::S`: target Metropolis-Hastings acceptance rate

**Other:**

- `n_presample_periods::Int`: number of initial periods to omit from the
  log-likelihood calculation
- `allout::Bool`: whether to return all outputs or just `sum(loglh)`
- `parallel::Bool`: whether to use `SharedArray`s
- `verbose::Symbol`: amount to print to STDOUT. One of `:none`, `:low`, or
  `:high`

### Outputs

- `sum(loglh)::S`: log-likelihood p(y_{1:T}|Φ,Ψ,F_ϵ,F_u) approximation
- `loglh::Vector{S}`: vector of conditional log-likelihoods p(y_t|y_{1:t-1},Φ,Ψ,F_ϵ,F_u)
- `times::Vector{S}`: vector of runtimes per period t
"""
## TODO: Allow Φ and Ψ to vary over time (do the same in TPF), just like in Kalman Filter.
function ensemble_kalman_filter(data::AbstractArray, Φ::Function, Ψ::Function,
                                  F_ϵ::Distribution, F_u::Distribution,
                                  s_init::AbstractArray{S}; n_particles::Int = 100,
                                  n_presample_periods::Int = 0, allout::Bool = true,
                                  parallel::Bool = false, get_t_particle_dist::Bool = false,
                                  verbose::Symbol = :high,
                                  dynamic_measurement::Bool = false) where S<:AbstractFloat

    #--------------------------------------------------------------
    # Setup
    #--------------------------------------------------------------

    # If using fixed φ schedule, check well-formed
    adaptive_φ = isempty(fixed_sched)
    if !adaptive_φ
        @assert fixed_sched[end] == 1 "φ schedule must be a range from [a,1] s.t. a > 0."
    end

    # Initialize constants
    n_obs, T  = size(data)
    n_shocks  = length(F_ϵ)
    n_states  = size(s_init, 1)
    QQerr = false
    HHerr = false
    try
        QQ = cov(F_ϵ)
    catch
        QQerr = true
    end
    try
        HH = cov(F_u)
    catch
        HHerr = true
    end
    if QQerr
        QQ = F_ϵ.σ * ones(1,1)
    end
    if HHerr
        HH = zeros(1,1)
    end
    @assert @isdefined HH

    # Initialize output vectors
    loglh = zeros(T)
    times = zeros(T)

    # Initialize matrix of normalized weight per particle by time period
    # and a matrix of the particle locations, if desired by user
    if get_t_particle_dist
        t_particle_dist = Dict{Int64,Matrix{Float64}}()
    end

    # Initialize working variables
    s_t1_temp     = Matrix{Float64}(copy(s_init))
    s_t_nontemp   = Matrix{Float64}(undef, n_states, n_particles)
    ϵ_t           = Matrix{Float64}(undef, n_shocks, n_particles)

    sendto(workers(), s_t1_temp = s_t1_temp)
    sendto(workers(), s_t_nontemp = s_t_nontemp)
    sendto(workers(), ϵ_t = ϵ_t)

    coeff_terms   = Vector{Float64}(undef, n_particles)
    log_e_1_terms = Vector{Float64}(undef, n_particles)
    log_e_2_terms = Vector{Float64}(undef, n_particles)

    inc_weights   = Vector{Float64}(undef, n_particles)
    norm_weights  = Vector{Float64}(undef, n_particles)

    c = c_init
    accept_rate = target_accept_rate

    #--------------------------------------------------------------
    # Main Algorithm: Tempered Particle Filter
    #--------------------------------------------------------------

    for t = 1:T
        begin_time = time_ns()
        if VERBOSITY[verbose] >= VERBOSITY[:low]
            println("============================================================")
            @show t
        end

        #--------------------------------------------------------------
        # Initialization
        #--------------------------------------------------------------

        y_t = data[:, t]

        # Remove rows/columns of series with NaN values
        # Handle measurement equation
        Ψ_t  = x -> Ψ(x)[nonmissing]
        sendto(workers(), Ψ_t = Ψ_t)


        # Adjust other values to remove rows/columns with NaN values
        nonmissing = isfinite.(y_t)
        y_t        = y_t[nonmissing]
        n_obs_t    = length(y_t)
        HH_t     = HH[nonmissing, nonmissing] # poolmodel -> keep missing is ok
        inv_HH_t = inv(HH_t) # poolmodel -> don't need inv_HH
        det_HH_t = det(HH_t) # poolmodel -> don't need det_HH

        # Initialize s_t_nontemp and ϵ_t for this period
        if parallel
            # Send to workers
            ϵ_t = rand(F_ϵ, n_particles)
            if ndims(ϵ_t) == 1 # Edge case where only 1 shock
                ϵ_t = reshape(ϵ_t, (1, length(ϵ_t)))
            end
            #sendto(workers(), s_t1_temp = s_t1_temp)
            #sendto(workers(), ϵ_t = ϵ_t)
            # sendto(workers(), Φ = Φ) ## Should be sent to all workers before calling the function

            state_transition_closure(i::Int) = Φ(s_t1_temp[:, i], ϵ_t[:, i])
            @everywhere state_transition_closure(i::Int) = Φ(s_t1_temp[:, i], ϵ_t[:, i])

            s_t_nontemp .= @sync @distributed (hcat) for i in 1:n_particles
                state_transition_closure(i)
            end
        else
            # Step 0: For later use
            avg_wts = fill(1.0 / n_particles, (n_particles, n_particles))
            update_prod = I - avg_wts

            # Step 1: Predict
            ##TODO: Need pseudoinverse for regular EnKF (not TEnKF)
            ##TODO: Do transposes if using TEnKF
             for i in 1:n_particles
                 ϵ_t[:, i] .= rand(F_ϵ)
                 s_t_nontemp[:, i] = Φ(s_t1_temp[:, i], ϵ_t[:, i])

                 Z_t_t1 = Ψ_t(s_t_nontemp[:,i]) + rand(F_u) ##TODO: Use Ψ_allstates as appropriate
                 ## This is different from econsieve and more in line with the equation
                 ## econsieve doesn't add measurement error to Z-bar.
                 ## In expectation, there is no difference.
            end

            # Step 2: Update
            Xbar = s_t_nontemp * update_prod
            Zbar = Z_t_t1 * update_prod
            Zcov = Zbar * Zbar'

            s_t_nontemp .+= Xbar * Zbar' * inv(Zcov) * (y_t .- Z_t_t1)

            # Step 3: Log Likelihood Update
            diff = y_t .- mean(Z_t_t1, dims = 1)
            loglh[t] = logpdf(MvNormal(Zcov), diff)#logpdf(x=y, mean=np.zeros(dim_z), cov=S)
        end

        if get_t_particle_dist
            t_particle_dist[t] = copy(s_t_nontemp)
        end

        times[t] = time_ns() - begin_time
    end

    if get_t_particle_dist && allout
        return sum(loglh[n_presample_periods + 1:end]), loglh[n_presample_periods + 1:end], times, t_particle_dist
    elseif get_t_particle_dist
        return sum(loglh[n_presample_periods + 1:end]), t_particle_dist
    elseif allout
        return sum(loglh[n_presample_periods + 1:end]), loglh[n_presample_periods + 1:end], times
    else
        return sum(loglh[n_presample_periods + 1:end])
    end

        # Tempering initialization
        φ_old = 1e-30
        stage = 0

        # Pass in objects that are only changed with iteration in t
        if parallel
            sendto(workers(), y_t = y_t)
            sendto(workers(), HH_t = HH_t)
            if !poolmodel
                sendto(workers(), inv_HH_t = inv_HH_t)
                #sendto(workers(), det_HH_t = det_HH_t)
            end
        end

        #--------------------------------------------------------------
        # Main Algorithm
        #--------------------------------------------------------------
        while φ_old < 1
            stage += 1

            ### 1. Correction
            # Modifies coeff_terms, log_e_1_terms, log_e_2_terms
            weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                           Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t;
                           initialize = stage == 1, parallel = parallel,
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

            ### 2. Selection
            # Modifies s_t1_temp, s_t_nontemp, ϵ_t
            selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t;
                       resampling_method = resampling_method)

            loglh[t] += log(mean(inc_weights))

            c = update_c(c, accept_rate, target_accept_rate)
            if VERBOSITY[verbose] >= VERBOSITY[:high]
                @show c
                println("------------------------------")
            end

            ### 3. Mutation
            # Modifies s_t_nontemp, ϵ_t
            if stage != 1
                accept_rate = mutation!(Φ, Ψ_t, QQ, det_HH_t, inv_HH_t, φ_new, y_t,
                                        s_t_nontemp, s_t1_temp, ϵ_t, c, n_mh_steps;
                                        parallel = parallel,
                                        poolmodel = poolmodel)
            end

            φ_old = φ_new
        end # of loop over stages

        if get_t_particle_dist
            # save the normalized weights in the column for period t
            t_norm_weights[:,t] .= norm_weights
            t_particle_dist[t] = copy(s_t_nontemp)
        end

        times[t] = time_ns() - begin_time
        if VERBOSITY[verbose] >= VERBOSITY[:low]
            print("\n")
            @show loglh[t]
            print("Completion of one period $times[t]")
        end
        s_t1_temp .= s_t_nontemp
    end # of loop over periods

    if VERBOSITY[verbose] >= VERBOSITY[:low]
        println("=============================================")
    end

    if get_t_particle_dist && allout
        return sum(loglh[n_presample_periods + 1:end]), loglh[n_presample_periods + 1:end], times, t_particle_dist, t_norm_weights
    elseif get_t_particle_dist
        return sum(loglh[n_presample_periods + 1:end]), t_particle_dist, t_norm_weights
    elseif allout
        return sum(loglh[n_presample_periods + 1:end]), loglh[n_presample_periods + 1:end], times
    else
        return sum(loglh[n_presample_periods + 1:end])
    end
end
