"""
```
bootstrap_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; verbose = :high,
    n_particles = 1000, findroot = bisection,
    xtol = 1e-3, resampling_method = :multionial, n_mh_steps = 1, c_init = 0.3,
    target_accept_rate = 0.4, n_presample_periods = 0, allout = true,
    parallel = false, verbose = :low)
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

- `findroot::Function`: root-finding algorithm, one of `bisection` or `fzero`
- `xtol::S`: root-finding tolerance on x bracket size

**Selection:**

- `resampling_method::Symbol`: either `:multinomial` or `:systematic`

**Mutation:**

- `n_mh_steps::Int`: number of Metropolis-Hastings steps to take
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
function bootstrap_particle_filter(data::Matrix{S}, Φ::Function, Ψ::Function,
                                  F_ϵ::Distribution, F_u::Distribution, s_init::Matrix{S};
                                  n_particles::Int = 1000, findroot::Function = bisection,
                                  xtol::S = 1e-3, resampling_method = :multinomial, n_mh_steps::Int = 1,
                                  c_init::S = 0.3, target_accept_rate::S = 0.4,
                                  n_presample_periods::Int = 0, allout::Bool = true,
                                  parallel::Bool = false, verbose::Symbol = :high) where S<:AbstractFloat
    #--------------------------------------------------------------
    # Setup
    #--------------------------------------------------------------

    # Initialize constants
    n_obs, T  = size(data)
    n_shocks  = length(F_ϵ)
    n_states  = size(s_init, 1)
    QQ        = cov(F_ϵ)
    HH        = cov(F_u)

    # Initialize output vectors
    loglh = zeros(T)
    times = zeros(T)

    # Initialize working variables
    MyVector = parallel ? SharedVector : Vector
    MyMatrix = parallel ? SharedMatrix : Matrix

    s_t1_temp     = MyMatrix{Float64}(copy(s_init))
    s_t_nontemp   = MyMatrix{Float64}(n_states, n_particles)
    ϵ_t           = MyMatrix{Float64}(n_shocks, n_particles)

    coeff_terms   = MyVector{Float64}(n_particles)
    log_e_terms = MyVector{Float64}(n_particles)

    inc_weights   = Vector{Float64}(n_particles)
    norm_weights  = Vector{Float64}(n_particles)

    c = c_init

    accept_rate = target_accept_rate

    #--------------------------------------------------------------
    # Main Algorithm: Bootstrap Particle Filter
    #--------------------------------------------------------------

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
        y_t = data[:, t]

        nonmissing = isfinite.(y_t)
        y_t        = y_t[nonmissing]
        n_obs_t    = length(y_t)
        Ψ_t        = x -> Ψ(x)[nonmissing]
        HH_t       = HH[nonmissing, nonmissing]
        inv_HH_t   = inv(HH_t)
        det_HH_t   = det(HH_t)

        # CHECK
        # NOTE
        # Initialize s_t_nontemp and ϵ_t for this period
        @mypar parallel for i in 1:n_particles
            ϵ_t[:, i] = rand(F_ϵ)
            s_t_nontemp[:, i] = Φ(@view(s_t1_temp[:, i]), @view(ϵ_t[:, i]))
        end

        #--------------------------------------------------------------
        # Main Algorithm
        #--------------------------------------------------------------

        ### 1. Correction
        # Modifies coeff_terms, log_e_terms
        weight_kernel!(coeff_terms, log_e_terms, Ψ, y_t, s_t_nontemp,
                       det_HH_t, inv_HH_t; parallel = parallel)

        # Modifies inc_weights, norm_weights
        correction!(inc_weights, norm_weights, coeff_terms,
                    log_e_terms, n_obs_t)

        ### 2. Selection
        # Modifies s_t1_temp, s_t_nontemp, ϵ_t
        selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t; resampling_method = resampling_method)

        loglh[t] += log(mean(inc_weights))

        c = update_c(c, accept_rate, target_accept_rate)
        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @show c
            println("------------------------------")
        end

        ### 3. Mutation

        # Modifies s_t_nontemp, ϵ_t

        accept_rate = mutation!(Φ, Ψ_t, QQ, det_HH_t, inv_HH_t, y_t, s_t_nontemp,
                                s_t1_temp, ϵ_t, c, n_mh_steps; parallel = parallel)


        if VERBOSITY[verbose] >= VERBOSITY[:low]
            print("\n")
            @show loglh[t]
            print("Completion of one period ")
            times[t] = toc()
        else
            times[t] = toq()
        end
        s_t1_temp .= s_t_nontemp
    end # of loop over periods

    if VERBOSITY[verbose] >= VERBOSITY[:low]
        println("=============================================")
    end

    if allout
        return sum(loglh[n_presample_periods + 1:end]), loglh[n_presample_periods + 1:end], times
    else
        return sum(loglh[n_presample_periods + 1:end])
    end
end
