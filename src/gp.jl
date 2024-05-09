# Make a "fake" AbstractGP type
const KroneckerProductGP = GP{<:MeanFunction, <:KernelKroneckerProduct}

# FiniteGP types 
const KroneckerFiniteGP = FiniteGP{<:KroneckerProductGP}

# If covariance is diagonal, must provide as a Kronecker product
const DiagonalVarKroneckerFiniteGP = FiniteGP{<:KroneckerProductGP, <:AbstractVecOrMat, <:Kronecker.KroneckerProduct{<:Real, <:Diagonal, <:Diagonal}}

# Assume a fill array means the variance is σ²I (maybe this isn't always true..)
const UniformVarKroneckerFiniteGP = FiniteGP{<:KroneckerProductGP, <:AbstractVecOrMat, <:Diagonal{<:Real, <:AbstractGPs.FillArrays.Fill}}

# For the case of Σ = σ^2 I
function Random.rand(rng::AbstractRNG, f::UniformVarKroneckerFiniteGP, N::Int)
    m = mean(f)
    K = kernelmatrix(f.f.kernel, f.x)
    E = eigen(K); λs = E.values; U = E.vectors
    Λ = Diagonal(λs) + f.Σy
    Λ12z = sqrt(Λ) * randn(rng, promote_type(eltype(m), eltype(Λ)), length(m), N)
    return m + (U * Λ12z)
end

# Case Σ is diagonal (can be slooooooow)
function Random.rand(rng::AbstractRNG, f::DiagonalVarKroneckerFiniteGP, N::Int)
    m = mean(f)
    K = kernelmatrix(f.f.kernel, f.x)
    E = eigen(K); Q = E.vectors; Λ1 = Diagonal(E.values)
    Λ = Λ1 + transpose(Q)*f.Σy*Q
    return m + Q*sqrt(Λ)*randn(rng, promote_type(eltype(m), eltype(Λ)), length(m), N)
end

Random.rand(f::KroneckerFiniteGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)
Random.rand(rng::AbstractRNG, f::KroneckerFiniteGP) = vec(rand(rng, f, 1))
Random.rand(f::KroneckerFiniteGP) = vec(rand(f, 1))

# Following Fortune et al. 2024
# This one assumes uniform input variance -> f.Σy == σ²I
function Distributions.logpdf(f::UniformVarKroneckerFiniteGP, Y::AbstractVecOrMat{<:Real})
    NM = size(Y,1)

    # Get the kronecker product matrix and compute the eigen decomposition of the components
    K = kernelmatrix(f.f.kernel, f.x)
    E1 = eigen(Symmetric(K.A)); Q1 = E1.vectors; Λ1 = Diagonal(E1.values)
    E2 = eigen(Symmetric(K.B)); Q2 = E2.vectors; Λ2 = Diagonal(E2.values)

    # Get the K^-1 * Y
    # Uses the "vec trick"
    σ2 = f.Σy[1]
    Λ = (Λ1 ⊗ Λ2) + σ2*I
    Q = (Q1 ⊗ Q2)
    invKY = Q * inv(Λ) * transpose(Q) * Y

    # Now get logdet of K
    logdetK = sum([log(Λ.diag[i]) for i in 1:NM])

    # Finally, compute logpdf
    return -0.5*(Y'invKY + logdetK + NM*log(2π))
end

# Now for Σ is diagonal
function Distributions.logpdf(f::DiagonalVarKroneckerFiniteGP, Y::AbstractVecOrMat{<:Real})
    NM = size(Y, 1)

    # Get kernel matrix
    K = kernelmatrix(f.f.kernel, f.x)

    # Get input covariance components
    Σ1 = f.Σy.A; Σ1m = inv(sqrt(Σ1))
    Σ2 = f.Σy.B; Σ2m = inv(sqrt(Σ2))
    Σm = Σ1m ⊗ Σ2m

    # Make transformed kernel components and get eigen decomposition
    K̃1 = Σ1m * K.A * Σ1m
    K̃2 = Σ2m * K.B * Σ2m
    E1 = eigen(Symmetric(K̃1)); Q1 = E1.vectors; Λ1 = Diagonal(E1.values)
    E2 = eigen(Symmetric(K̃2)); Q2 = E2.vectors; Λ2 = Diagonal(E2.values)

    # Compute invKY
    Λ = (Λ1 ⊗ Λ2) + I
    invKY = Σm * (Q1 ⊗ Q2) * (inv(Λ) * (transpose(Q1) ⊗ transpose(Q2)) * Σm * Y)

    # Now logdetK
    logdetK = sum([log(Λ.diag[i]) + log(f.Σy[i,i]) for i in 1:NM])
    
    return -0.5*(Y'invKY + logdetK + NM*log(2π))
end
