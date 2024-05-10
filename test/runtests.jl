using AbstractGPs, KernelFunctions, KroneckerProductKernels, Kronecker
using Test, BenchmarkTools, Random
using KernelFunctions: TestUtils
using KroneckerProductKernels: make_test_inputs
using Kronecker: ⊗, Diagonal

@testset "KroneckerProductKernels.jl" begin
    include("test_kernel.jl")
    include("test_gp.jl")
end
