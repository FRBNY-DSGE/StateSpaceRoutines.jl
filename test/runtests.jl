using StateSpaceRoutines
using Test, HDF5, JLD2, FileIO, LinearAlgebra, PDMats, Distributions, Random

my_tests = [
            "kalman_filter",
            "chand_recursion",
            "tempered_particle_filter",
            "smoothers",
            "dyn_measure_tpf",
            "EnKF"
            ]

for test in my_tests
    @testset "$test.jl" begin
        include(string("$test.jl"))
    end
end
