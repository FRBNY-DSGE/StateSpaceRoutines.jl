using Base.Test, HDF5, JLD
using StateSpaceRoutines
using BenchmarkTools
using QuantEcon: solve_discrete_lyapunov

my_tests = [
            "kalman_filter"
            "chand_recursion"
            "tempered_particle_filter"
            "smoothers"
            ]

for test in my_tests
    test_file = string("$test.jl")
    @testset "$test.jl" begin
        #@printf " * %s\n" test_file
        include(test_file)
    end
end
