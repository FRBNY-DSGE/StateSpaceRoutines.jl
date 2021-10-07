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
                                  poolmodel::Bool = false) where S<:AbstractFloat
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
        t_norm_weights = Matrix{Float64}(undef,n_particles,T)
        t_particle_dist = Dict{Int64,Matrix{Float64}}()
    end

    # Initialize working variables
#=
    s_t1_temp     = parallel ? SharedMatrix{Float64}(copy(s_init)) :
        Matrix{Float64}(copy(s_init))
    s_t_nontemp   = parallel ? SharedMatrix{Float64}(n_states, n_particles) :
        Matrix{Float64}(undef, n_states, n_particles)
    ϵ_t           = parallel ? SharedMatrix{Float64}(n_shocks, n_particles) :
        Matrix{Float64}(undef, n_shocks, n_particles)
    coeff_terms   = parallel ? SharedVector{Float64}(n_particles) :
        Vector{Float64}(undef, n_particles)
    log_e_1_terms = parallel ? SharedVector{Float64}(n_particles) :
        Vector{Float64}(undef, n_particles)
    log_e_2_terms = parallel ? SharedVector{Float64}(n_particles) :
        Vector{Float64}(undef, n_particles)
=#
    s_t1_temp     = parallel ? distribute(copy(s_init)) :
        Matrix{Float64}(copy(s_init))
    s_t_nontemp   = parallel ? dzeros(n_states, n_particles) :
        Matrix{Float64}(undef, n_states, n_particles)
    ϵ_t           = parallel ? dzeros(n_shocks, n_particles) :
        Matrix{Float64}(undef, n_shocks, n_particles)
    coeff_terms   = parallel ? dzeros(n_particles) :
        Vector{Float64}(undef, n_particles)
    log_e_1_terms = parallel ? dzeros(n_particles) :
        Vector{Float64}(undef, n_particles)
    log_e_2_terms = parallel ? dzeros(n_particles) :
        Vector{Float64}(undef, n_particles)

    inc_weights   = parallel ? dones(n_particles) : Vector{Float64}(undef, n_particles)
    norm_weights  = parallel ? dones(n_particles) : Vector{Float64}(undef, n_particles)

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

    ## Store unnormalized weights for resample with processors when BSPF in parallel
    if parallel && fixed_sched == [1.0]
        unnormalized_wts = dones(n_particles)
        processor_wt = dones(nworkers())
    end

    for t = 1:T
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
        if parallel
            if n_shocks == 1 && ndims(rand(F_ϵ,1)) == 1
                ϵ_t = DArray((n_shocks, n_particles)) do inds
                    reshape(rand(F_ϵ, length(inds[1])), (1, length(inds[1])))
                end
            else
                ϵ_t = DArray((n_shocks,n_particles), workers(), [1,nworkers()]) do inds
                    rand(F_ϵ, length(inds[2]))
                end
            end

            if ndims(ϵ_t) == 1 # Edge case where only 1 shock
                ϵ_t = reshape(ϵ_t, (1, length(ϵ_t)))
            end
            # Cite for DArray: https://discourse.julialang.org/t/alternative-to-sharedarrays-for-multi-node-cluster/37794
            s_t_nontemp = DArray((n_states,n_particles), workers(), [1,nworkers()]) do inds # the 1 in dist = [1,x] important because each chunk should have complete columns
                arr = zeros(inds) # OffsetArray corrects local indices
                s_t1 = OffsetArray(localpart(s_t1_temp),DistributedArrays.localindices(s_t1_temp)) # worker also stores some part of s_t1_temp which this gets and corrects
                for i in inds[2]
                    arr[:,i] .= Φ(convert(Vector,s_t1[:, i]), ϵ_t[:, i]) # Need to convert for type-checking in Φ
                end
                parent(arr) # remove the OffsetArray wrapper
            end
        else
             for i in 1:n_particles
                ϵ_t[:, i] .= rand(F_ϵ)
                s_t_nontemp[:, i] = Φ(s_t1_temp[:, i], ϵ_t[:, i])
            end
        end

        # Tempering initialization
        φ_old = 1e-30
        stage = 0

        # Distribute particles across processors in each iteration of below loop
        ## Use DistributedArrays.SPMD to run whol thing in parallel
        ## 1. Construct everything as a DArray - DONE Earlier
        ## 1.5 Check dimension division is the same for each object
        ### (each worker has all particles for each object)
        ## 2. Write below while loop as a function
        ## 3. Use SPMD to run in parallel
        ## 4. Resample at the end of each iteration following Gust et al.

        #--------------------------------------------------------------
        # Main Algorithm
        #--------------------------------------------------------------
        while φ_old < 1
            stage += 1

            if parallel
                if fixed_sched == [1.0] ## TPF would be faster with mutation at start and adaptive TPF can't be parallelized like this
                    φ_old = spmd(tpf_helper!, coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                            Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                            n_obs_t, stage, inc_weights, norm_weights,
                            s_t1_temp, ϵ_t,
                            Φ, Ψ_t, QQ, det_HH_t, inv_HH_t,
                            r_star = r_star,
                            initialize = stage == 1, poolmodel = poolmodel,
                            fixed_sched = fixed_sched,
                            findroot = findroot, xtol = xtol,
                            resampling_method = resampling_method, target_accept_rate = target_accept_rate,
                            accept_rate = accept_rate, n_mh_steps = n_mh_steps,
                            verbose = verbose; pids=workers())

                # Resample processors if necessary
                    unnormalized_wts[:L] = unnormalized_wts[:L] .* inc_weights[:L]
                    processor_wt[:L] = sum(unnormalized_wts[:L])
                    procs_wt2 = convert(Vector,processor_wt)
                    alpha_k = processor_wt2 ./ sum(processor_wt2)
                    alpha_args = sortperm(alpha_k)
                    EP_t = 1/sum(alpha_k .^ 2)

                    if EP_t < nworkers()/2
                        replace_inds = n_particles ÷ (2*nworkers())
                        for i in 1:floor(nworkers()/2)
                            first_proc = alpha_args[i]
                            last_proc = alpha_args[nworkers()+1-i]

                            # Either take each DArray and change its indices directly
                            ## or change the localparts on each processor to match.
                            ### Let's do the former unless it doesn't change what's on each processor.
                            #### Hmmm, indexing into DArrays is hard but easy for localparts so let's do the latter.

                            tmp1 = @fetchfrom first_proc localpart(unnormalized_weights)
                            tmp2 = @fetchfrom last_proc localpart(unnormalized_weights)
                            remotecall_fetch(set_dvals, last_proc, unnormalized_weights, 1:replace_inds, tmp1[1:replace_inds])
                            remotecall_fetch(set_dvals, first_proc, unnormalized_weights, 1:replace_inds, tmp2[1:replace_inds])

                            tmp1 = @fetchfrom first_proc localpart(s_t1_temp)
                            tmp2 = @fetchfrom last_proc localpart(s_t1_temp)
                            remotecall_fetch(set_dvals, last_proc, s_t1_temp, 1:replace_inds, tmp1[1:replace_inds])
                            remotecall_fetch(set_dvals, first_proc, s_t1_temp, 1:replace_inds, tmp2[1:replace_inds])

                            tmp1 = @fetchfrom first_proc localpart(s_t_nontemp)
                            tmp2 = @fetchfrom last_proc localpart(s_t_nontemp)
                            remotecall_fetch(set_dvals, last_proc, s_t_nontemp, 1:replace_inds, tmp1[1:replace_inds])
                            remotecall_fetch(set_dvals, first_proc, s_t_nontemp, 1:replace_inds, tmp2[1:replace_inds])

                            ## ϵ_t only needs to be stored when tempering
                            #=tmp1 = @fetchfrom first_proc localpart(ϵ_t)
                            tmp2 = @fetchfrom last_proc localpart(ϵ_t)
                            remotecall_fetch(set_dvals, last_proc, ϵ_t, 1:replace_inds, tmp1[1:replace_inds])
                            remotecall_fetch(set_dvals, first_proc, ϵ_t, 1:replace_inds, tmp2[1:replace_inds])=#

                            ## coeff_terms, log_e_1_terms, and log_e_2_terms are reset in the next iteration.
                            #=tmp1 = @fetchfrom first_proc localpart(coeff_terms)
                            tmp2 = @fetchfrom last_proc localpart(coeff_terms)
                            remotecall_fetch(set_dvals, last_proc, coeff_terms, 1:replace_inds, tmp1[1:replace_inds])
                            remotecall_fetch(set_dvals, first_proc, coeff_terms, 1:replace_inds, tmp2[1:replace_inds])

                            tmp1 = @fetchfrom first_proc localpart(log_e_1_terms)
                            tmp2 = @fetchfrom last_proc localpart(log_e_1_terms)
                            remotecall_fetch(set_dvals, last_proc, log_e_1_terms, 1:replace_inds, tmp1[1:replace_inds])
                            remotecall_fetch(set_dvals, first_proc, log_e_1_terms, 1:replace_inds, tmp2[1:replace_inds])

                            tmp1 = @fetchfrom first_proc localpart(log_e_2_terms)
                            tmp2 = @fetchfrom last_proc localpart(log_e_2_terms)
                            remotecall_fetch(set_dvals, last_proc, log_e_2_terms, 1:replace_inds, tmp1[1:replace_inds])
                            remotecall_fetch(set_dvals, first_proc, log_e_2_terms, 1:replace_inds, tmp2[1:replace_inds])=#

                            ## inc_weights is reset and only the mean is used in the iteration
                            #=tmp1 = @fetchfrom first_proc localpart(s_t_nontemp)
                            tmp2 = @fetchfrom last_proc localpart(s_t_nontemp)
                            remotecall_fetch(set_dvals, last_proc, s_t_nontemp, 1:replace_inds, tmp1[1:replace_inds])
                            remotecall_fetch(set_dvals, first_proc, s_t_nontemp, 1:replace_inds, tmp2[1:replace_inds])=#

                            tmp1 = @fetchfrom first_proc localpart(norm_weights)
                            tmp2 = @fetchfrom last_proc localpart(norm_weights)
                            remotecall_fetch(set_dvals, last_proc, norm_weights, 1:replace_inds, tmp1[1:replace_inds])
                            remotecall_fetch(set_dvals, first_proc, norm_weights, 1:replace_inds, tmp2[1:replace_inds])
                        end
                        @distributed for i in workers()
                            processor_wt[:L] = sum(unnormalized_wts[:L])
                        end
                    end

                # Update loglikelihood
                loglh[t] += log(mean(inc_weights))
                end
            else
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
                if parallel
                    s_t_nontemp_std = convert(Array, s_t_nontemp)
                    s_t1_temp_std = convert(Array, s_t1_temp)

                    selection!(norm_weights, s_t1_temp_std, s_t_nontemp_std, ϵ_t;
                               resampling_method = resampling_method)

                    s_t_nontemp = distribute(s_t_nontemp_std)
                    s_t1_temp = distribute(s_t1_temp_std)
                else
                    selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t;
                               resampling_method = resampling_method)
                end

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
            end
        end # of loop over stages

        if get_t_particle_dist
            # save the normalized weights in the column for period t
            t_norm_weights[:,t] = parallel ? convert(Array, norm_weights) : norm_weights
            t_particle_dist[t] = parallel ? convert(Array, s_t_nontemp) : copy(s_t_nontemp)
        end

        times[t] = time_ns() - begin_time
        if VERBOSITY[verbose] >= VERBOSITY[:low]
            print("\n")
            @show loglh[t]
            print("Completion of one period $times[t]")
        end
        if parallel
            s_t1_temp = s_t_nontemp
        else
            s_t1_temp .= s_t_nontemp
        end
    end # of loop over periods

    if VERBOSITY[verbose] >= VERBOSITY[:low]
        println("=============================================")
    end

    if parallel
        d_closeall()
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


# Function for parallel Bootstrap Particle Filter
        function tpf_helper!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                             Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t,
                             n_obs_t, stage, inc_weights, norm_weights,
                             s_t1_temp, ϵ_t,
                             Φ, Ψ_t, QQ, det_HH_t, inv_HH_t;
                             r_star::S = 2.0,
                             initialize = 1, poolmodel::Bool = false,
                             fixed_sched::Vector{S} = zeros(0),
                             findroot::Function = bisection, xtol::S = 1e-3,
                             resampling_method::Symbol = :multinomial, target_accept_rate::S = 0.4,
                             accept_rate::S = target_accept_rate, n_mh_steps::Int = 1,
                             verbose::Symbol = :high) where S<:AbstractFloat

            ### 1. Correction
            # Modifies coeff_terms, log_e_1_terms, log_e_2_terms
            weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old,
                           Ψ_allstates, y_t, s_t_nontemp, det_HH_t, inv_HH_t;
                           initialize = stage == 1, parallel = true,
                           poolmodel = poolmodel)

            φ_new = fixed_sched[stage] ## Function only runs w/ Bootstrap PF so this is 1.0
            #=φ_new = next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs_t,
                           r_star, stage; fixed_sched = fixed_sched, findroot = findroot,
                           xtol = xtol, parallel = parallel)=#

            if VERBOSITY[verbose] >= VERBOSITY[:high]
                @show φ_new
            end

            # Modifies inc_weights, norm_weights
            correction!(inc_weights, norm_weights, φ_new, coeff_terms,
                        log_e_1_terms, log_e_2_terms, n_obs_t, parallel = parallel)

            ### 2. Selection
            # Modifies s_t1_temp, s_t_nontemp, ϵ_t
            selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t;
                           resampling_method = resampling_method, parallel = parallel)

            c = update_c(c, accept_rate, target_accept_rate)
            if VERBOSITY[verbose] >= VERBOSITY[:high]
                @show c
                println("------------------------------")
            end

            ### 3. Mutation
            # Modifies s_t_nontemp, ϵ_t
            if stage != 1 ## Note this never runs in Bootstrap PF case
                accept_rate = mutation!(Φ, Ψ_t, QQ, det_HH_t, inv_HH_t, φ_new, y_t,
                                        s_t_nontemp, s_t1_temp, ϵ_t, c, n_mh_steps;
                                        parallel = parallel,
                                        poolmodel = poolmodel)
            end

            φ_old = φ_new

            return φ_old
        end

## Set values on a specific processor
@everywhere function set_dvals(x,inds,final_val)
    x[:L][inds] .= final_val
end
