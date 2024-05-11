# Extend KernelFunctions 

#### Types and constructors ####
struct KernelKroneckerProduct{K} <: KernelFunctions.Kernel
    kernels::K
end
KernelKroneckerProduct(k1::K1, k2::K2) where {K1, K2} = KernelKroneckerProduct((k1,k2))

#### properties ####
Base.length(kernel::KernelKroneckerProduct) = length(kernel.kernels)

function Base.:(==)(x::KernelKroneckerProduct, y::KernelKroneckerProduct)
    return (
        length(x) == length(y) &&
        all(kx == ky for (kx, ky) in zip(x.kernels, y.kernels))
    )
end

#### Kernel matrix #####
# Base on KernelTensorProduct
function (kernel::KernelKroneckerProduct)(x, y)
    if !(length(x) == length(y) == length(kernel))
        throw(DimensionMismatch("number of kernels and number of features are not consistent"))
    end
    return prod(k(xi, yi) for (k, xi, yi) in zip(kernel.kernels, x, y))
end

function validate_domain(k::KernelKroneckerProduct, x::AbstractVector)
    return KernelFunctions.dim(x) == length(k) ||
        error("number of kernels and groups of features are not consistent")
end

# Inputs should be all the pairs sorted like: [[x y] for x in x for y in y]
# Each first dimension input must be paired with each of a fix set of second
# dimension inputs. Eg. [x1, y1], ...,[x1, yn], [x2, y1], ..., [x2, yn]
function KernelFunctions.kernelmatrix(k::KernelKroneckerProduct, x::AbstractVector)
    validate_domain(k, x)
    x1, x2 = unique.(KernelFunctions.slices(x)) # Get the unique entries in each dimension
    K1 = kernelmatrix(k.kernels[1], x1) # Build both independent kernel matrices
    K2 = kernelmatrix(k.kernels[2], x2)
    return K1 ⊗ K2 # Return a Kronecker product matrix type
end

function KernelFunctions.kernelmatrix(k::KernelKroneckerProduct, x::AbstractVector, y::AbstractVector)
    validate_domain(k, x)
    validate_domain(k, y)
    x1, x2 = unique.(KernelFunctions.slices(x))
    y1, y2 = unique.(KernelFunctions.slices(y))
    K1 = kernelmatrix(k.kernels[1], x1, y1)
    K2 = kernelmatrix(k.kernels[2], x2, y2)
    return K1 ⊗ K2
end

function KernelFunctions.kernelmatrix_diag(k::KernelKroneckerProduct, x::AbstractVector)
    K = kernelmatrix(k, x)
    return Diagonal(K).diag
end