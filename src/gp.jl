# Implement FiniteGP interface
const KroneckerFiniteGP = AbstractGPs.FiniteGP{<:GP{<:MeanFunction, <:KernelKroneckerProduct}}

# For the case of Σ + σ^2 I
function Random.rand(rng::AbstractRNG, f::KroneckerFiniteGP, N::Int)
    m = mean(f)
    K = kernelmatrix(f.f.kernel, f.x)
    E = eigen(K); λs = E.values; U = E.vectors
    Λ = Diagonal(λs) .+ f.Σy
    Λ12z = sqrt(Λ) * randn(rng, promote_type(eltype(m), eltype(Λ)), length(m), N)
    return m + (U * Λ12z)
end
Random.rand(f::KroneckerFiniteGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)
Random.rand(rng::AbstractRNG, f::KroneckerFiniteGP) = vec(rand(rng, f, 1))
Random.rand(f::KroneckerFiniteGP) = vec(rand(f, 1))

# Following Fortune et al. 2024
# This one assumes uniform white noise (make this explicit later)
function Distributions.logpdf(f::KroneckerFiniteGP, Y::AbstractVecOrMat{<:Real})
    N = size(Y,1)
    # Get the kronecker product matrix
    K = kernelmatrix(f.f.kernel, f.x)
    # Compute the eigen decomposition of each component matrix
    E1 = eigen(Symmetric(K.A)); Q1 = E1.vectors; Λ1 = Diagonal(E1.values)
    E2 = eigen(Symmetric(K.B)); Q2 = E2.vectors; Λ2 = Diagonal(E2.values)

    # Get the K^-1 * Y
    Λ1Λ2 = Λ1 ⊗ Λ2
    invKY = (Q1 ⊗ Q2) * inv(Λ1Λ2 + f.Σy) * (transpose(Q1) ⊗ transpose(Q2)) * Y

    # Not get logdet of K
    logdetK = sum([log(Λ1Λ2.diag[i] + f.Σy.diag[i]) for i in 1:N])

    return -0.5*(Y'invKY + logdetK + N*log(2π))
end