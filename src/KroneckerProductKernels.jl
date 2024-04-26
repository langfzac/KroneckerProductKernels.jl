module KroneckerProductKernels

using AbstractGPs, KernelFunctions, Kronecker
using Distributions
using LinearAlgebra
using Kronecker: ⊗

export KernelKroneckerProduct

include("kernel.jl")

end
