using Test, HDF5, JLD2, Printf, LinearAlgebra, Statistics, FileIO, PDMats, Distributions, Random
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
    @testset "$test.jl" begin
        include(string("$test.jl"))
    end
end
