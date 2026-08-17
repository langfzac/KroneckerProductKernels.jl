module KroneckerProductKernelsForwardDiffExt
    using ForwardDiff, Kronecker, LinearAlgebra
    using ForwardDiff: Dual, Partials, value, partials
    using Kronecker: ⊗ 
    import KroneckerProductKernels

    # Compute quasi-analytic gradient using the method from Fortune et al. 2024
    # This bypasses the need to differentiate the eigen decomposition
    function KroneckerProductKernels._logpdf(
        f::KroneckerProductKernels.UniformVarKroneckerFiniteGP, 
        Y::AbstractVecOrMat{<:Real}, 
        K::Kronecker.KroneckerProduct{D}
        ) where D <: Dual
        T = ForwardDiff.tagtype(D)
        Np = ForwardDiff.npartials(D)
        NM = size(Y,1)

        # Extract primals for the Kronecker matrices
        K1 = value.(K.A)
        K2 = value.(K.B)

        # Get the kronecker product matrix and compute the eigen decomposition of the components
        E1 = eigen(Symmetric(K1)); Q1 = E1.vectors; Λ1 = Diagonal(E1.values)
        E2 = eigen(Symmetric(K2)); Q2 = E2.vectors; Λ2 = Diagonal(E2.values)

        # Get the K^-1 * Y
        # Uses the "vec trick"
        σ2 = value(f.Σy[1])
        Λ = (Λ1 ⊗ Λ2) + σ2*I
        invΛ = inv(Λ)
        Q = (Q1 ⊗ Q2)
        Qt = Q'
        invKY = Q * invΛ * Qt * Y
        invKYt = invKY'

        # logpdf primal
        logdetK = sum(log.(Λ.diag))
        logpdf_val = -0.5 * (Y'invKY + logdetK + NM*log(2π))

        # Compute gradients for each partial
        grads = ntuple(Val(Np)) do i 
            dK1 = partials.(K.A, i)
            dK2 = partials.(K.B, i)
            dΣ = partials.(f.Σy[1], i)

            # Compute the quadratic term 
            quad = invKYt * ((dK1 ⊗ K2) * invKY) + invKYt * ((K1 ⊗ dK2) * invKY)
            
            # Now the log determinant term
            dlogdetK = dot(diag(invΛ), (diag(Q1' * dK1 * Q1) ⊗ diag(Q2' * K2 * Q2))) + dot(diag(invΛ), (diag(Q1' * K1 * Q1) ⊗ diag(Q2' * dK2 * Q2)))

            # Handle the white noise parameter
            if dΣ != 0.0
                quad += dot(invKYt, invKY) * dΣ
                dlogdetK += tr(invΛ) * dΣ
            end

            0.5 * (quad - dlogdetK)
        end
        return Dual{T}(logpdf_val, Partials(grads))
    end
end