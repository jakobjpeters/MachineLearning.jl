
const Assessment = @NamedTuple{accuracies, costs}

struct Dataset{A1<:AbstractArray{Float32}, A2<:AbstractArray{Float32}}
    x::A1
    y::A2
end

function Dataset(x, y)
    return Dataset(convert.(Float32, x), convert.(Float32, y))
end

# corresponds to a layer in a 'Neural_Network'
struct LayerParameters{F1, F2, F3}
    normalize::F1
    activate::F2
    regularize::F3
    λ::Float32 # regularizer rate
    η::Float32 # learning rate
end

function LayerParameters(normalize, activate, regularize, λ, η)
    λ = convert(Float32, λ)
    η = convert(Float32, η)

    return LayerParameters(normalize, activate, regularize, λ, η)
end

struct EpochParameters{T<:Integer, F1, F2, VL<:AbstractVector{<:LayerParameters}}
    batch_size::T
    shuffle::Bool
    loss::F1
    normalize::F2
    layers_params::VL
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

struct Linear{S<:Union{AbstractVector{Float32}, Float32}, R<:Union{Float32, Nothing}} <: Model
    w::S
    b::R
end

# not implemented yet
struct GenerativeAdversarialNetwork{T1<:Model, T2<:Model} <: Model
    generator::T1
    discriminator::T2
end
