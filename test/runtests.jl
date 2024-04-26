using KroneckerProductKernels, KernelFunctions
using Test, Random
using KernelFunctions: TestUtils

@testset "KroneckerProductKernels.jl" begin
    include("test_kernel.jl")
end
