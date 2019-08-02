using StateSpaceRoutines, Test, HDF5, JLD2, FileIO, LinearAlgebra, PDMats, Distributions, Random, DSGE

my_tests = [
            "kalman_filter"
            "chand_recursion"
            "tempered_particle_filter"
            "smoothers"
            "dyn_measure_tpf"
            "poolmodel_tpf"
            ]

for test in my_tests
    @testset "$test.jl" begin
        include(string("$test.jl"))
    end
end
