# Function for testing different sized inputs 
function make_test_inputs(Nx, Ny=nothing; seed=1234)
    if Ny === nothing; Ny = Nx; end
    rng = Random.MersenneTwister(seed)
    xs = collect(1:1:Nx) .+ 0.1*randn(rng, Nx)
    ys = collect(1:1:Ny) .+ 0.1*randn(rng, Ny)
    return vcat([[x y] for x in xs for y in ys]...)
end