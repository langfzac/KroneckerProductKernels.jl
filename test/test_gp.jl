@testset "GP" begin
    @testset "logpdf" begin
        # Test the logpdfs for a few combinations of kernels
        # Again, the computations for the tensor product kernel should match
        X = make_test_inputs(100,50)
        kernels = (ExponentialKernel(), SqExponentialKernel(), Matern52Kernel(), WhiteKernel())
        for k1 in kernels
            for k2 in kernels
                k_kron = KernelKroneckerProduct(k1, k2)
                k_tensor = KernelTensorProduct(k1, k2)

                # Get FiniteGP for each
                sigma = 0.1
                fx_kron = GP(k_kron)(RowVecs(X), sigma)
                fx_tensor = GP(k_tensor)(RowVecs(X), sigma)
                
                # draw from finite gp
                Y = rand(fx_kron)

                # Compare logpdfs 
                @test logpdf(fx_kron, Y) ≈ logpdf(fx_tensor, Y)

                # Check type stability
                @test @inferred(logpdf(fx_kron, Y)) isa Float64

                # Report benchmarks
                #=@info k_kron.kernels
                println("Kronecker: ")
                @btime logpdf($fx_kron, $Y)
                println("Tensor: ")
                @btime logpdf($fx_tensor, $Y)=#
            end
        end
    end
end