# KroneckerProductKernels

[![Build Status](https://github.com/langfzac/KroneckerProductKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/langfzac/KroneckerProductKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/langfzac/KroneckerProductKernels.jl/graph/badge.svg?token=TMWM3I40O9)](https://codecov.io/gh/langfzac/KroneckerProductKernels.jl)

A Julia package for computing 2D GPs with Kronecker product kernels. Based on [`luas`](https://github.com/markfortune/luas) from [Fortune et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240215204F/abstract). This package uses the [`KernelFunctions.jl`](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) and [`AbstractGPs.jl`](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) interfaces, which works with inference tools such as [`Turing.jl`](https://github.com/TuringLang/Turing.jl). Kernel matrix evaluations rely on [`Kronecker.jl`](https://github.com/MichielStock/Kronecker.jl) for fast computation and low-memory overhead. 

# Example

```julia
using AbstractGPs, KernelFunction, KroneckerProductKernels
using LinearAlgebra, Random; Random.seed!(1234)
using Kronecker: ⊗ # \otimes<tab>

# Create a KernelKroneckerProduct using standard KernelFunctions kernels
k1 = 0.1 * Matern52Kernel()
k2 = SqExponentialKernel() ∘ ScaleTransform(0.4) # \circ<tab> -> ∘
kernel = KernelKroneckerProduct(k1, k2)

# And get an AbstractGP
f = GP(kernel)

# Inputs must be structured in a particular way
# Need a set of 2nd dimension values for every 1st dimension value
# Inputs should be sorted by the first dimension
X1 = collect(0.0:1:100)
X2 = collect(0.0:1:5)
X = vcat([[x1 x2] for x1 in X1 for x2 in X2]...)

# We can now get a FiniteGP
# Either with a uniform, diagonal covariance
fx = f(RowVecs(X), 0.1)

# Or with a non-uniform, diagonal covariance
# This must be provided as a KroneckerProduct
C1 = Diagonal(rand(length(X1)))
C2 = Diagonal(rand(length(X2)))
C = C1 ⊗ C2
fx = f(RowVecs(X), C)

# Now, we can draw from the GP
Y = rand(fx)

# And evaluate the logpdf
logpdf(fx, Y)
```