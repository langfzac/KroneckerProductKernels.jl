@testset "GP" begin
    # Redo some tests from AbstractGPs
    @testset "convert" begin
        N,M = 10, 5
        X = make_test_inputs(N, M)
        covariances = (0.1, Diagonal(rand(N)) ⊗ Diagonal(rand(M)))
        for covar in covariances
            fx = GP(KernelKroneckerProduct(0.1 * SqExponentialKernel(), SqExponentialKernel()))(RowVecs(X), covar)
            fx_scaled = GP(0.1 * KernelKroneckerProduct(SqExponentialKernel(), SqExponentialKernel()))(RowVecs(X), covar)
            @test fx isa AbstractGPs.FiniteGP
            
            dist = @inferred(convert(MvNormal, fx))
            dist_scaled = @inferred(convert(MvNormal, fx_scaled))
            @test dist isa MvNormal{Float64}
            @test dist ≈ dist_scaled
            @test mean(dist) ≈ mean(fx)
            @test cov(dist) ≈ cov(fx)
        end
    end

    @testset "rand" begin
        # Test that rand is functioning as expected
        rng = Random.MersenneTwister(1234)
        N, M = 10, 5
        X = make_test_inputs(N, M)
        kernel = KernelKroneckerProduct(SqExponentialKernel(), 0.1 * SqExponentialKernel())
        kernel_scaled = 0.1 * KernelKroneckerProduct(SqExponentialKernel(), SqExponentialKernel())
        
        # Check for uniform and diagonal covariance
        covariances = (0.1, Diagonal(rand(N)) ⊗ Diagonal(rand(M)))
        for covariance in covariances
            fx = GP(kernel)(RowVecs(X), covariance)
            fx_scaled = GP(kernel_scaled)(RowVecs(X), covariance)
            dist = convert(MvNormal, fx)

            # Make sure rand methods work
            @test length(rand(fx, 1)) == size(X,1)
            @test size(rand(rng, fx, 10)) == (size(X,1), 10)
            @test length(rand(fx)) == size(X,1)
            @test size(rand(fx, 10)) == (size(X, 1), 10)

            # Make sure returns real vector
            @test rand(rng, fx) isa Vector{Float64}
            @test rand(rng, fx_scaled) isa Vector{Float64}

            # Check that samples converge to a MvNormal
            ns = 1_000_000
            f̂1 = rand(rng, fx, ns)
            f̂2 = rand(rng, fx, ns)
            f̂3 = rand(rng, fx_scaled, ns)
            f̂4 = rand(rng, fx_scaled, ns)
            for f̂ in (f̂1, f̂2, f̂3, f̂4)
                @test maximum(abs.(mean(f̂; dims=2) - mean(fx))) < 1e-2

                res = (f̂ .- mean(fx))
                Σp = res * res' ./ ns 
                @test mean(abs.(Σp - cov(fx))) < 1e-2
            end
        end
    end

    @testset "logpdf" begin
        # Test the logpdfs for a few combinations of kernels
        # Again, the computations for the tensor product kernel should match
        N, M = 100, 5
        X = make_test_inputs(N,M)
        kernels = (ExponentialKernel(), SqExponentialKernel(), Matern52Kernel(), WhiteKernel())
        for k1 in kernels
            for k2 in kernels
                k_kron = KernelKroneckerProduct(k1, 0.1 * k2)
                k_kron_scaled = 0.1 * KernelKroneckerProduct(k1, k2)
                k_tensor = 0.1 * KernelTensorProduct(k1, k2)

                # Test both uniform and diagonal covariance
                covariances = (0.1, Diagonal(rand(N)) ⊗ Diagonal(rand(M)))
                for covariance in covariances
                    fx_kron = GP(k_kron)(RowVecs(X), covariance)
                    fx_kron_scaled = GP(k_kron_scaled)(RowVecs(X), covariance)
                    fx_tensor = GP(k_tensor)(RowVecs(X), covariance)
                
                    # draw from finite gp
                    Y = rand(fx_kron)

                    # Compare logpdfs 
                    @test logpdf(fx_kron, Y) ≈ logpdf(fx_tensor, Y)
                    @test logpdf(fx_kron, Y) ≈ logpdf(fx_kron_scaled, Y)

                    # Check type stability
                    @test @inferred(logpdf(fx_kron, Y)) isa Float64
                    @test @inferred(logpdf(fx_kron_scaled, Y)) isa Float64

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