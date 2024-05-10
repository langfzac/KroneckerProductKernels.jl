@testset "GP" begin
    @testset "logpdf" begin
        # Test the logpdfs for a few combinations of kernels
        # Again, the computations for the tensor product kernel should match
        N, M = 100, 5
        X = make_test_inputs(N,M)
        kernels = (ExponentialKernel(), SqExponentialKernel(), Matern52Kernel(), WhiteKernel())
        for k1 in kernels
            for k2 in kernels
                k_kron = KernelKroneckerProduct(k1, k2)
                k_tensor = KernelTensorProduct(k1, k2)

                # Test both uniform and diagonal covariance
                covariances = (0.1, Diagonal(rand(N)) ⊗ Diagonal(rand(M)))
                for covariance in covariances
                    fx_kron = GP(k_kron)(RowVecs(X), covariance)
                    fx_tensor = GP(k_tensor)(RowVecs(X), covariance)
                
                    # draw from finite gp
                    Y = rand(fx_kron)

                    # Compare logpdfs 
                    @test logpdf(fx_kron, Y) ≈ logpdf(fx_tensor, Y)

                    # Check type stability
                    @test @inferred(logpdf(fx_kron, Y)) isa Float64

                    # Report benchmarks
                    #=@info k_kron.kernels covariance
                    println("Kronecker: ")
                    @btime logpdf($fx_kron, $Y)
                    println("Tensor: ")
                    @btime logpdf($fx_tensor, $Y)=#
                end
            end
        end
    end
end