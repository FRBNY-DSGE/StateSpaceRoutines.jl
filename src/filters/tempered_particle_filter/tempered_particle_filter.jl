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
- `s_init::Array{S}`: `Ns` x `n_particles` matrix of initial state vectors s_0

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
- `parallel::Bool`: whether to use `DistributedArray`s
- `verbose::Symbol`: amount to print to STDOUT. One of `:none`, `:low`, or
  `:high`
- `parallel_testing::Bool`: Set Random.seeds if testing parallel implementation

### Outputs

- `sum(loglh)::S`: log-likelihood p(y_{1:T}|Φ,Ψ,F_ϵ,F_u) approximation
- `loglh::Vector{S}`: vector of conditional log-likelihoods p(y_t|y_{1:t-1},Φ,Ψ,F_ϵ,F_u)
- `times::Vector{S}`: vector of runtimes per period t
"""
function tempered_particle_filter(data::AbstractArray, Φ::Function, Ψ::Function,
                                  F_ϵ::Distribution, F_u::Distribution,
                                  s_init::AbstractArray{S}; n_particles::Int = 1000,
                                  fixed_sched::Vector{S} = zeros(0), r_star::S = 2.0,
                                  findroot::Function = bisection, xtol::S = 1e-3,
                                  resampling_method = :multinomial, n_mh_steps::Int = 1,
                                  c_init::S = 0.3, target_accept_rate::S = 0.4,
                                  n_presample_periods::Int = 0, allout::Bool = true,
                                  parallel::Bool = false, get_t_particle_dist::Bool = false,
                                  verbose::Symbol = :high,
                                  dynamic_measurement::Bool = false,
                                  poolmodel::Bool = false, parallel_testing::Bool = false) where S<:AbstractFloat

    # Run the main helper function which will do TPF.
    ## This splits between parallel and non-parallel
    ## implementations for type stability.
    return if parallel
        parallel_tempered_particle_filter(data, Φ, Ψ,
                                          F_ϵ, F_u,
                                          s_init; n_particles = n_particles, fixed_sched, r_star,
                                          findroot, xtol,
                                          resampling_method, n_mh_steps,
                                          c_init, target_accept_rate,
                                          n_presample_periods, allout,
                                          get_t_particle_dist,
                                          verbose, dynamic_measurement,
                                          poolmodel, parallel_testing)
    else
        sequential_tempered_particle_filter(data, Φ, Ψ,
                                            F_ϵ, F_u,
                                            s_init; n_particles = n_particles, fixed_sched = fixed_sched,
                                            r_star = r_star, findroot = findroot,
                                            xtol = xtol, resampling_method = resampling_method,
                                            n_mh_steps = n_mh_steps, c_init = c_init,
                                            target_accept_rate = target_accept_rate,
                                            n_presample_periods = n_presample_periods, allout = allout,
                                            get_t_particle_dist = get_t_particle_dist,
                                            verbose = verbose, dynamic_measurement = dynamic_measurement,
                                            poolmodel = poolmodel, parallel_testing = parallel_testing)
    end
end


# Sequential tempered_particle_filter
function sequential_tempered_particle_filter(data::AbstractArray, Φ::Function, Ψ::Function,
                                             F_ϵ::Distribution, F_u::Distribution,
                                             s_init::AbstractArray{S}; n_particles::Int = 1000,
                                             fixed_sched::Vector{S} = zeros(0), r_star::S = 2.0,
                                             findroot::Function = bisection, xtol::S = 1e-3,
                                             resampling_method = :multinomial, n_mh_steps::Int = 1,
                                             c_init::S = 0.3, target_accept_rate::S = 0.4,
                                             n_presample_periods::Int = 0, allout::Bool = true,
                                             get_t_particle_dist::Bool = false,
                                             verbose::Symbol = :high,
                                             dynamic_measurement::Bool = false,
                                             poolmodel::Bool = false, parallel_testing::Bool = false) where S<:AbstractFloat

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

    if length(size(s_init)) == 1
        n_states = 1
    end

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
        QQ = var(F_ϵ) * ones(1,1)
    end
    if HHerr
        HH = try
            var(F_u) .* ones(1,1)
        catch
            zeros(1,1)
        end
    end
    @assert @isdefined HH

    # Initialize output vectors
    loglh = zeros(T)
    times = zeros(T)

    # Initialize matrix of normalized weight per particle by time period
    # and a matrix of the particle locations, if desired by user
    if get_t_particle_dist
        t_norm_weights = Matrix{Float64}(undef,n_particles,T)
        t_particle_dist = Dict{Int64,Matrix{Float64}}()
    end

    # Initialize working variables
    # Note: Vector used when n_states == 1 (assumed that n_shocks <= 1 then too)
    ## TODO: Use vector for ϵ_t when n_shocks = 1, n_states > 1
    s_t1_temp     = (n_states == 1 && n_shocks == 1) ? Vector{Float64}(copy(vec(s_init))) :
        (ndims(s_init) > 1 ? Matrix{Float64}(copy(s_init)) : Matrix{Float64}(reshape(copy(s_init),(1,length(s_init)))))
    s_t_nontemp   = (n_states == 1 && n_shocks == 1) ? Vector{Float64}(undef, n_particles) :
        Matrix{Float64}(undef, n_states, n_particles)
    ϵ_t           = (n_states == 1 && n_shocks == 1) ? Vector{Float64}(undef, n_particles) :
        Matrix{Float64}(undef, n_shocks, n_particles)

    coeff_terms   = Vector{Float64}(undef, n_particles)
    log_e_1_terms = Vector{Float64}(undef, n_particles)
    log_e_2_terms = Vector{Float64}(undef, n_particles)

    inc_weights   = Vector{Float64}(undef, n_particles)
    norm_weights  = Vector{Float64}(undef, n_particles)

    c = c_init
    accept_rate = target_accept_rate

    # If not using a dynamic measurement equation, then define measurement equation
    # applying to all states, even if they are missing (but not time variables)
    if !dynamic_measurement
        Ψ_allstates = Ψ
    end

    #--------------------------------------------------------------
    # Main Algorithm: Tempered Particle Filter
    #--------------------------------------------------------------

    for t = 1:T
        if parallel_testing
            Random.seed!(t)
        end
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
        if !(poolmodel || dynamic_measurement)
            Ψ_t  = x -> Ψ(x)[nonmissing]
        elseif poolmodel && dynamic_measurement
            Ψ_t = x -> Ψ(x,y_t,t)[nonmissing]
            Ψ_allstates = x -> Ψ(x,y_t,t)
        elseif poolmodel
            Ψ_t = x -> Ψ(x,y_t)[nonmissing]
            Ψ_allstates = x -> Ψ(x,y_t)
        else
            Ψ_t  = x -> Ψ(x,t)[nonmissing]
            Ψ_allstates = x -> Ψ(x,t)
        end

        # Adjust other values to remove rows/columns with NaN values
        nonmissing = isfinite.(y_t)
        y_t        = y_t[nonmissing]
        n_obs_t    = length(y_t)
        HH_t     = poolmodel ? HH : HH[nonmissing, nonmissing] # poolmodel -> keep missing is ok
        inv_HH_t = poolmodel ? zeros(1,1) : inv(HH_t) # poolmodel -> don't need inv_HH
        det_HH_t = poolmodel ? 0. : det(HH_t) # poolmodel -> don't need det_HH

        # Initialize s_t_nontemp and ϵ_t for this period
        if n_states > 1 || n_shocks > 1
            for i in 1:n_particles
                ϵ_t[:,i] .= rand(F_ϵ)
                s_t_nontemp[:,i] .= Φ(s_t1_temp[:,i], ϵ_t[:,i])
            end
        else
            ϵ_t = rand(F_ϵ, n_particles)
            s_t_nontemp = Φ(s_t1_temp, ϵ_t)
        end

        # Tempering initialization
        φ_old = 1e-30
        stage = 0

        if parallel_testing
            testings = 0
            Random.seed!(testings)
        end

        #--------------------------------------------------------------
        # Main Algorithm
        #--------------------------------------------------------------

        while φ_old < 1
            stage += 1

            if parallel_testing
                testings += 5
                Random.seed!(testings)
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

            ### 2. Selection
            # Modifies s_t1_temp, s_t_nontemp, ϵ_t
            if fixed_sched == [1.0]
                ## Only need to resample s_t_nontemp when no mutation b/c rest reset in next time iteration.
                selection!(norm_weights, s_t_nontemp;
                           resampling_method = resampling_method)
            else
                selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t;
                           resampling_method = resampling_method)
            end

            loglh[t] += log(mean(inc_weights))

            ### 3. Mutation
            # Modifies s_t_nontemp, ϵ_t
            if stage != 1
                c = update_c(c, accept_rate, target_accept_rate)
                if VERBOSITY[verbose] >= VERBOSITY[:high]
                    @show c
                    println("------------------------------")
                end

                accept_rate = mutation!(Φ, Ψ_t, QQ, det_HH_t, inv_HH_t, φ_new, y_t,
                                        s_t_nontemp, s_t1_temp, ϵ_t, c, n_mh_steps;
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

# Parallel tempered particle_filter
function parallel_tempered_particle_filter(data::AbstractArray, Φ::Function, Ψ::Function,
                                           F_ϵ::Distribution, F_u::Distribution,
                                           s_init::AbstractArray{S}; n_particles::Int = 1000,
                                           fixed_sched::Vector{S} = zeros(0), r_star::S = 2.0,
                                           findroot::Function = bisection, xtol::S = 1e-3,
                                           resampling_method = :multinomial, n_mh_steps::Int = 1,
                                           c_init::S = 0.3, target_accept_rate::S = 0.4,
                                           n_presample_periods::Int = 0, allout::Bool = true,
                                           get_t_particle_dist::Bool = false,
                                           verbose::Symbol = :high,
                                           dynamic_measurement::Bool = false,
                                           poolmodel::Bool = false, parallel_testing::Bool = false) where S<:AbstractFloat

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

    if length(size(s_init)) == 1
        n_states = 1
    end

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
        QQ = var(F_ϵ) * ones(1,1)
    end
    if HHerr
        HH = try
            var(F_u) .* ones(1,1)
        catch
            zeros(1,1)
        end
    end

    # Initialize output vectors
    loglh = zeros(T)
    times = zeros(T)

    # Initialize matrix of normalized weight per particle by time period
    # and a matrix of the particle locations, if desired by user
    if get_t_particle_dist
        t_norm_weights = Matrix{Float64}(undef,n_particles,T)
        t_particle_dist = Dict{Int64,Matrix{Float64}}()
    end

    # Initialize working variables
    # TODO: Ensure each worker has all of a particle
    # Note: Vector used when n_states == 1 (assumed that n_shocks <= 1 then too)
    ## TODO: Use vector for ϵ_t when n_shocks = 1, n_states > 1
    s_t1_temp     = (n_states == 1 && n_shocks == 1) ? distribute(copy(vec(s_init))) :
        (ndims(s_init) > 1 ? distribute(copy(s_init), dist = [1, nworkers()]) :
        distribute(reshape(copy(s_init), (1,length(s_init))), dist = [1, nworkers()]))
    s_t_nontemp   = (n_states == 1 && n_shocks == 1) ? dzeros(n_particles) :
        dzeros((n_states, n_particles), workers(), [1,nworkers()])
    ϵ_t           = (n_states == 1 && n_shocks == 1) ? dzeros(n_particles) :
        dzeros((n_shocks, n_particles), workers(), [1,nworkers()])

    coeff_terms   = dzeros(n_particles)
    log_e_1_terms = dzeros(n_particles)
    log_e_2_terms = dzeros(n_particles)

    inc_weights   = dones(n_particles)
    norm_weights  = dones(n_particles)

    ## TODO: assert that localindices for each worker is the same across all DArrays

    c_vec = dfill(c_init, nworkers())
    accept_rate = dfill(target_accept_rate, nworkers())

    # If not using a dynamic measurement equation, then define measurement equation
    # applying to all states, even if they are missing (but not time variables)
    if !dynamic_measurement
        Ψ_allstates = Ψ
    end

    #--------------------------------------------------------------
    # Main Algorithm: Tempered Particle Filter
    #--------------------------------------------------------------

    ## Store unnormalized weights for resample with processors when BSPF in parallel
    unnormalized_wts = dones(n_particles)

    for t = 1:T
        if parallel_testing
            Random.seed!(t)
        end
        begin_time = time_ns()
        if VERBOSITY[verbose] >= VERBOSITY[:low]
            println("============================================================")
            @show t
        end

        #--------------------------------------------------------------
        # Initialization
        #--------------------------------------------------------------

        # Remove rows/columns of series with NaN values
        y_t = data[:, t]

        # Handle measurement equation
        if !(poolmodel || dynamic_measurement)
            Ψ_t  = x -> Ψ(x)[nonmissing]
        elseif poolmodel && dynamic_measurement
            Ψ_t = x -> Ψ(x,y_t,t)[nonmissing]
            Ψ_allstates = x -> Ψ(x,y_t,t)
        elseif poolmodel
            Ψ_t = x -> Ψ(x,y_t)[nonmissing]
            Ψ_allstates = x -> Ψ(x,y_t)
        else
            Ψ_t  = x -> Ψ(x,t)[nonmissing]
            Ψ_allstates = x -> Ψ(x,t)
        end

        # Adjust other values to remove rows/columns with NaN values
        nonmissing = isfinite.(y_t)
        y_t        = y_t[nonmissing]
        n_obs_t    = length(y_t)
        HH_t     = poolmodel ? HH : HH[nonmissing, nonmissing] # poolmodel -> keep missing is ok
        inv_HH_t = poolmodel ? zeros(1,1) : inv(HH_t) # poolmodel -> don't need inv_HH
        det_HH_t = poolmodel ? 0. : det(HH_t) # poolmodel -> don't need det_HH

        # Initialize s_t_nontemp and ϵ_t for this period
        ϵ_t_vec = rand(F_ϵ, n_particles)

        if n_states == 1 && n_shocks == 1
            ϵ_t = distribute(ϵ_t_vec)

            @sync @distributed for w in workers()
                s_t_nontemp[:L][:] .= Φ.(s_t1_temp[:L], ϵ_t[:L])
            end
        else
            if n_shocks == 1 # Edge case where only 1 shock
                ϵ_t_vec = reshape(ϵ_t_vec, (1, length(ϵ_t_vec)))
            end
            ϵ_t = distribute(ϵ_t_vec, dist = [1, nworkers()])

            @sync @distributed for w in workers()
                for i in 1:size(s_t_nontemp[:L],2)
                    s_t_nontemp[:L][:,i] = Φ(s_t1_temp[:L][:,i], ϵ_t[:L][:,i])
                end
            end
        end

        # Tempering initialization
        φ_old = 1e-30
        stage = 0

        if parallel_testing
            @everywhere testings = 0
            @everywhere Random.seed!(testings)
        end

        if fixed_sched != [1.0]
            stage += 1
            if parallel_testing
                @everywhere testings += 5
                @everywhere Random.seed!(testings)
            end

            if adaptive_φ
                spmd(adaptive_weight_kernel!, coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                     Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                     stage == 1, poolmodel; pids=workers())

                φ_new = next_φ(φ_old, convert(Vector, coeff_terms),
                               convert(Vector, log_e_1_terms), convert(Vector, log_e_2_terms),
                               n_obs_t, r_star, stage;
                               fixed_sched = fixed_sched, findroot = findroot, xtol = xtol)

                if VERBOSITY[verbose] >= VERBOSITY[:high]
                    @show φ_new
                end

                # Correction + update unnormalized_wts
                spmd(adaptive_correction!, inc_weights, norm_weights, φ_new, coeff_terms,
                     log_e_1_terms, log_e_2_terms, n_obs_t, unnormalized_wts, s_t1_temp, s_t_nontemp, ϵ_t, resampling_method; pids=workers())

                φ_old = φ_new
            else
                spmd(one_iter!, coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                     Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                     n_obs_t, stage, inc_weights, norm_weights,
                     s_t1_temp, ϵ_t, unnormalized_wts,
                     r_star, poolmodel,
                     fixed_sched, findroot, xtol,
                     resampling_method, n_mh_steps,
                     verbose; pids=workers())
                φ_old = fixed_sched[stage]
            end

            # Resample processors if necessary
            if nworkers() == 1
                EP_t = 1.0
            else
                procs_wt = @sync @distributed (vcat) for p in workers()
                    unnormalized_wts[:L][1]
                end

                α_k = procs_wt ./ sum(procs_wt)
                EP_t = 1/sum(α_k .^ 2)
            end

            if EP_t < nworkers()/2
                unnormalized_wts_vec = convert(Vector, unnormalized_wts)

                s_t1_temp_vec = convert(Array, s_t1_temp)
                s_t_nontemp_vec = convert(Array, s_t_nontemp)
                ϵ_t_vec = convert(Array, ϵ_t)

                selection!(unnormalized_wts_vec, s_t1_temp_vec, s_t_nontemp_vec, ϵ_t_vec)

                if ndims(s_t_nontemp_vec) > 1
                    s_t1_temp = distribute(s_t1_temp_vec, dist = [1, nworkers()])
                    s_t_nontemp = distribute(s_t_nontemp_vec, dist = [1, nworkers()])
                    ϵ_t = distribute(ϵ_t_vec, dist = [1, nworkers()])
                else
                    s_t1_temp = distribute(s_t1_temp_vec)
                    s_t_nontemp = distribute(s_t_nontemp_vec)
                    ϵ_t = distribute(ϵ_t_vec)
                end
                unnormalized_wts = dones(length(unnormalized_wts_vec))
            end
        end

        #--------------------------------------------------------------
        # Main Algorithm
        #--------------------------------------------------------------

        while φ_old < 1 || fixed_sched != [1.0]
            stage += 1

            if fixed_sched == [1.0] ## TPF would be faster with mutation at start and adaptive TPF can't be parallelized like this
                if parallel_testing
                    @everywhere testings += 5
                    @everywhere Random.seed!(testings)
                end

                spmd(tpf_helper!, coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                     Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                     n_obs_t, stage, inc_weights, norm_weights,
                     s_t1_temp, ϵ_t, unnormalized_wts,
                     poolmodel,
                     resampling_method,
                     verbose; pids=workers())
                φ_old = 1.0 ## Can do this because it's given in fixed_sched

                # Update loglikelihood
                loglh[t] += log(mean(convert(Vector, inc_weights)))
            else
                # Parallel but adaptive schedule or fixed schedule but not bootstrap
                ## Same as BSPF with different loop order.
                ## First, correction, selection.
                ## Then start loop w/ mutation, corection, then recalculate φ and resample
                stage -= 1

                # Calculating log likelihood after each run
                loglh[t] += log(mean(convert(Vector, inc_weights)))

                ### 3. Mutation
                # Modifies s_t_nontemp, ϵ_t
                if φ_old >= 1
                    if stage != 1
                        @sync @distributed for p in workers()
                            c_vec[:L][1] = update_c(c_vec[:L][1], accept_rate[:L][1], target_accept_rate)

                            if VERBOSITY[verbose] >= VERBOSITY[:high]
                                @show c_vec
                                println("------------------------------")
                            end # Reset immediately in the next time iteration

                            accept_rate[:L][1] = mutation!(Φ, Ψ_t, QQ, det_HH_t, inv_HH_t, φ_old, y_t,
                                                           s_t_nontemp, s_t1_temp, ϵ_t, c_vec[:L][1], n_mh_steps;
                                                           poolmodel = poolmodel)
                        end
                        ## Don't need to use c_init b/c stage = 1 then.
                    end
                    break
                end
                stage += 1

                if adaptive_φ
                    spmd(adaptive_tempered_iter!, coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                         Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                         s_t1_temp, ϵ_t, c_vec,
                         Φ, Ψ_t, QQ, stage, accept_rate,
                         poolmodel, target_accept_rate,
                         n_mh_steps,
                         verbose; pids=workers())

                    if parallel_testing
                        @everywhere testings += 5
                        @everywhere Random.seed!(testings)
                    end

                    φ_new = next_φ(φ_old, convert(Vector, coeff_terms),
                                   convert(Vector, log_e_1_terms), convert(Vector, log_e_2_terms),
                                   n_obs_t, r_star, stage;
                                   fixed_sched = fixed_sched, findroot = findroot, xtol = xtol)

                    # Correction + update unnormalized_wts
                    spmd(adaptive_correction!, inc_weights, norm_weights, φ_new, coeff_terms,
                         log_e_1_terms, log_e_2_terms, n_obs_t, unnormalized_wts, s_t1_temp, s_t_nontemp, ϵ_t, resampling_method; pids=workers())

                    φ_old = φ_new
                else
                    if !parallel_testing
                        spmd(tempered_iter!, coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                             Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                             n_obs_t, stage, inc_weights, norm_weights,
                             s_t1_temp, ϵ_t, c_vec,
                             Φ, Ψ_t, QQ, unnormalized_wts,
                             accept_rate, r_star, poolmodel,
                             fixed_sched, findroot, xtol,
                             resampling_method, target_accept_rate,
                             n_mh_steps,
                             verbose; pids=workers())
                    else
                        spmd(tempered_iter_test!, coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                             Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                             n_obs_t, stage, inc_weights, norm_weights,
                             s_t1_temp, ϵ_t, c_vec,
                             Φ, Ψ_t, QQ,
                             accept_rate, r_star, poolmodel,
                             fixed_sched, findroot, xtol,
                             target_accept_rate,
                             n_mh_steps,
                             verbose; pids=workers())

                        @everywhere testings += 5
                        @everywhere Random.seed!(testings)

                        spmd(selection_test!, norm_weights, s_t1_temp, s_t_nontemp, ϵ_t,
                             unnormalized_wts, inc_weights, resampling_method)
                    end
                    φ_old = fixed_sched[stage]
                end
            end

            # Resample processors if necessary
            EP_t = if nworkers() == 1
                1.0
            else
                procs_wt = @sync @distributed (vcat) for p in workers()
                    unnormalized_wts[:L][1]
                end

                α_k = procs_wt ./ sum(procs_wt)
                1/sum(α_k .^ 2)
            end

            if EP_t < nworkers()/2
                unnormalized_wts_vec = convert(Vector, unnormalized_wts)
                ## unnormalized_wts_vec should be divided by mean regularly when this condition not called
                ### to avoid a particle being degenerate. However, since we resample at each step
                ### this won't be a problem as long as this step is called enough.

                s_t1_temp_vec = convert(Array, s_t1_temp)
                s_t_nontemp_vec = convert(Array, s_t_nontemp)
                ϵ_t_vec = convert(Array, ϵ_t)

                selection!(unnormalized_wts_vec, s_t1_temp_vec, s_t_nontemp_vec, ϵ_t_vec)

                if ndims(s_t_nontemp_vec) > 1
                    s_t1_temp = distribute(s_t1_temp_vec, dist = [1, nworkers()])
                    s_t_nontemp = distribute(s_t_nontemp_vec, dist = [1, nworkers()])
                    ϵ_t = distribute(ϵ_t_vec, dist = [1, nworkers()])
                else
                    s_t1_temp = distribute(s_t1_temp_vec)
                    s_t_nontemp = distribute(s_t_nontemp_vec)
                    ϵ_t = distribute(ϵ_t_vec)
                end
                unnormalized_wts = dones(length(unnormalized_wts_vec))
            end
        end # of loop over stages

        if get_t_particle_dist
            # save the normalized weights in the column for period t
            t_norm_weights[:,t] = convert(Array, norm_weights)
            t_particle_dist[t] = convert(Array, s_t_nontemp)
        end

        times[t] = time_ns() - begin_time
        if VERBOSITY[verbose] >= VERBOSITY[:low]
            print("\n")
            @show loglh[t]
            print("Completion of one period $times[t]")
        end

        @sync @distributed for p in workers()
            s_t1_temp[:L][:] .= s_t_nontemp[:L][:]
        end
    end # of loop over periods

    if VERBOSITY[verbose] >= VERBOSITY[:low]
        println("=============================================")
    end

    d_closeall()

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
