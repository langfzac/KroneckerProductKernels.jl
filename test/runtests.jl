using KroneckerProductKernels, KernelFunctions, AbstractGPs, BenchmarkTools
using Test, Random
using KernelFunctions: TestUtils
using KroneckerProductKernels: make_test_inputs

@testset "KroneckerProductKernels.jl" begin
    include("test_kernel.jl")
    include("test_gp.jl")
end
