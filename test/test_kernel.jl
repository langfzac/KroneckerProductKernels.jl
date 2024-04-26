@testset "kernel" begin
    # Follow tests in KernelFunctions.jl
    rng = MersenneTwister(1234)
    u1 = rand(rng, 10)
    u2 = rand(rng, 10)
    v1 = rand(rng, 5)
    v2 = rand(rng, 5)

    # Get base kernels 
    k1 = SqExponentialKernel()
    k2 = ExponentialKernel()

    # Check the constructor
    K1 = KernelKroneckerProduct(k1,k2)
    K2 = KernelKroneckerProduct([k1,k2])
    @test K1 == K2
    @test K1.kernels === (k1, k2) === KernelKroneckerProduct((k1,k2)).kernels

    # Test properties
    @test length(K1) == length(K2) == 2
    @test_throws DimensionMismatch K1(rand(3), rand(2))

    # Kronecker product is a special case of tensor product. The kernel 
    # computations should match here
    K1_tensor = KernelTensorProduct(k1,k2)

    # Make sure the kernel evaluations return the same value
    @testset "val" begin
        for (x ,y) in (((v1, u1), (v2, u2)), ([v1, u1], [v2, u2]))
            val = k1(x[1], y[1]) * k2(x[2], y[2])

            @test K1(x, y) == K2(x, y) == val
            @test K1(x, y) == K1_tensor(x, y)
        end
    end

    # To compute the kernelmatrix correctly, the first dimension must be paired
    # with each of the second dimension values. The second dimension must have
    # a fixed set of values. eg. [t1, λ1], ...,[t1, λn], [t2, λ1], ..., [t2, λn]
    X0 = make_test_inputs(5,4)
    X1 = make_test_inputs(5,4,seed=12345)
    X2 = make_test_inputs(6, 4)

    # Check that Kronecker and tensor kernel matrices match (not exactly, since
    # using distinct algorithms)
    @test kernelmatrix(K1, RowVecs(X0)) ≈ kernelmatrix(K1_tensor, RowVecs(X0))
    @test kernelmatrix(K1, RowVecs(X0), RowVecs(X1)) ≈ kernelmatrix(K1_tensor, RowVecs(X0), RowVecs(X1))
    @test kernelmatrix(K1, RowVecs(X0), RowVecs(X2)) ≈ kernelmatrix(K1_tensor, RowVecs(X0), RowVecs(X2))
    @test kernelmatrix(K1, ColVecs(X0')) ≈ kernelmatrix(K1_tensor, ColVecs(X0'))
    @test kernelmatrix(K1, ColVecs(X0'), ColVecs(X1')) ≈ kernelmatrix(K1_tensor, ColVecs(X0'), ColVecs(X1'))
    @test kernelmatrix(K1, ColVecs(X0'), ColVecs(X2')) ≈ kernelmatrix(K1_tensor, ColVecs(X0'), ColVecs(X2'))

    # Run standard interface tests
    TestUtils.test_interface(K1, RowVecs(X0), RowVecs(X1), RowVecs(X2))
    TestUtils.test_interface(K1, ColVecs(X0'), ColVecs(X1'), ColVecs(X2'))
    TestUtils.test_type_stability(K1, RowVecs(X0), RowVecs(X1), RowVecs(X2))
    TestUtils.test_type_stability(K1, ColVecs(X0'), ColVecs(X1'), ColVecs(X2'))
end