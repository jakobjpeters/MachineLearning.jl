
struct Dataset{A1<:AbstractArray{Float32}, A2<:AbstractArray{Float32}}
    x::A1
    y::A2
    n::Int64
end

# TODO: make internal constructor
function Dataset(x::T, y::T) where T<:AbstractArray{Float32}
    N = length(size(x))
    size(x, N) == size(y, N) || throw(ErrorException("Arguments are not the same length"))
    return Dataset(x, y, size(x, N))
end

function Dataset(x, y)
    return Dataset(convert.(Float32, x), convert.(Float32, y))
end

struct Regularizer{F}
    regularize::F
    λ::Float32
end

function Regularizer(regularize, λ)
    return Regularizer(regularize, convert(Float32, λ))
end

# corresponds to layers in a 'Neural_Network'
struct LayersParameters{F1<:AbstractArray, F2<:AbstractArray, R<:AbstractArray{<:Regularizer}, T<:AbstractArray{Float32}}
    normalizers::F1
    activators::F2
    regularizers::R
    η::T # learning rate
end

function LayersParameters(normalizers::AbstractArray, activators::AbstractArray, regularizers::AbstractArray, η::AbstractArray)
    return LayersParameters(normalizers, activators, regularizers, convert.(Float32, η))
end

# TODO: make generated function
function LayersParameters(normalizers, activators, regularizers, η, n_layers)
    if !isa(normalizers, AbstractArray)
        normalizers = repeat([normalizers], n_layers)
    end
    if !isa(activators, AbstractArray)
        activators = repeat([activators], n_layers)
    end
    if !isa(regularizers, AbstractArray)
        regularizers = repeat([regularizers], n_layers)
    end
    if !isa(η, AbstractArray)
        η = repeat([η], n_layers)
    end

    return LayersParameters(normalizers, activators, regularizers, η)
end

function LayersParameters(normalizers, activators, η)
    regularizer = Regularizer(weight_decay, 0.0)
    return LayersParameters(normalizers, activators, regularizer, η)
end

struct EpochParameters{T<:Integer, F1, F2, LP<:LayersParameters}
    batch_size::T
    shuffle::Bool
    loss::F1
    normalize::F2
    layers_params::LP
end

# corresponds to a layer in a 'Neural_Network'
mutable struct Cache{M<:AbstractMatrix{Float32}}
    δe_δl::M # δ error / δ linear
    l::M # linear
    a::M # activation
end

function Cache()
    init = Matrix{Float32}(undef, 0, 0)
    n_fields = length(fieldnames(Cache))

    return Cache(repeat([init], n_fields)...)
end

abstract type Layer end

# functor, see 'core.jl'
mutable struct Dense{M<:AbstractMatrix{Float32}, VN<:Union{AbstractVector{Float32}, Nothing}} <: Layer
    w::M # weight
    b::VN # bias
end

function Dense(init_w, x_size, size, use_bias)
    w = convert.(Float32, rand(init_w(x_size), size, x_size))
    b = use_bias ? zeros(Float32, size) : nothing

    return Dense(w, b)
end

abstract type Model end

# functor, see 'core.jl'
struct NeuralNetwork{VL<:AbstractVector{<:Layer}} <: Model
    layers::VL
end

function NeuralNetwork(x_size, w_inits, sizes, use_biases)
    x_sizes = pushfirst!(sizes[begin:end - 1], x_size)
    layers_params = zip(w_inits, x_sizes, sizes, use_biases)
    layers = map(layer_params -> Dense(layer_params...), layers_params)

    return NeuralNetwork(layers)
end

mutable struct Linear{S<:Union{AbstractVector{Float32}, Float32}, R<:Union{Float32, Nothing}} <: Model
    w::S
    b::R
end

function Linear(w, b = nothing)
    b = isnothing(b) ? nothing : convert(Float32, b)
    return Linear(convert.(Float32, w), b)
end

function Linear(use_bias::Bool = true)
    bias = use_bias ? 0.0 : nothing
    return Linear(0.0, bias)
end

# not implemented yet
struct GenerativeAdversarialNetwork{T1<:Model, T2<:Model} <: Model
    generator::T1
    discriminator::T2
end
