module KroneckerProductKernels

using AbstractGPs, KernelFunctions, Kronecker
using LinearAlgebra, Random

using Kronecker: ⊗

import AbstractGPs: MeanFunction, FiniteGP
import Distributions, Random

export KernelKroneckerProduct

include("kernel.jl")
include("gp.jl")

end
