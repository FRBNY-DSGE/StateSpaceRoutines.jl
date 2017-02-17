module StateSpaceRoutines

    using QuantEcon: solve_discrete_lyapunov

    export

        # kalman_filter.jl
        kalman_filter,

        # smoothers/
        hamilton_smoother, koopman_smoother, koopman_disturbance_smoother, carter_kohn_smoother, durbin_koopman_smoother

    include("kalman_filter.jl")

    include("smoothers/util.jl")
    include("smoothers/hamilton_smoother.jl")
    include("smoothers/koopman_smoother.jl")
    include("smoothers/carter_kohn_smoother.jl")
    include("smoothers/durbin_koopman_smoother.jl")

end