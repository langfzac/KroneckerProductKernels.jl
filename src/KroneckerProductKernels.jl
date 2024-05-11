module KroneckerProductKernels

using AbstractGPs, KernelFunctions, Kronecker
using LinearAlgebra, Random

using Kronecker: ⊗

import AbstractGPs: AbstractGP, MeanFunction, FiniteGP
import Distributions, Random

export KernelKroneckerProduct, KroneckerFiniteGP

include("kernel.jl")
include("gp.jl")
include("utils.jl")

end
