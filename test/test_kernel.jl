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
    x0s = collect(0.0:1:5); x1s = x0s .+ 5; x2s = collect(0.0:1:6)
    y0s = collect(1.0:1:5); y1s = y0s .+ 3; y2s = collect(1.0:1:6)
    X0 = vcat([[x y] for x in x0s for y in y0s]...)
    X1 = vcat([[x y] for x in x1s for y in y1s]...)
    X2 = vcat([[x y] for x in x2s for y in y2s]...)

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