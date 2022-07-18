
const Assessment = @NamedTuple{accuracies, costs}

struct Dataset{T<:AbstractFloat, A1<:AbstractArray{T}, A2<:AbstractArray{T}}
    x::A1
    y::A2
end

function Dataset(x, y, precision = Float32)
    return Data(convert.(precision, x), convert.(precision, y)) # convert?
end

# corresponds to a layer in a 'Neural_Network'
struct LayerParameters{F1, F2, F3, T<:AbstractFloat}
    normalize::F1
    activate::F2
    regularize::F3
    λ::T # regularizer rate
    η::T # learning rate
end

function LayerParameters(normalize, activate, regularize, λ, η, precision = Float32)
    return LayerParameters(normalize, activate, regularize, convert.(precision, λ), convert.(precision, η)) # convert?
end

struct EpochParameters{T<:Integer, F1, F2, VL<:AbstractVector{<:LayerParameters}}
    batch_size::T
    shuffle::Bool
    loss::F1
    normalize::F2
    layers_params::VL
end

# corresponds to a layer in a 'Neural_Network'
mutable struct Cache{T<:AbstractFloat, M<:AbstractMatrix{T}}
    δe_δl::M # δ error / δ linear
    l::M # linear
    a::M # activation
end

function Cache(precision = Float32)
    return Cache(repeat([Matrix{precision}(undef, 0, 0)], length(fieldnames(Cache)))...)
end

abstract type Layer end

# functor, see 'core.jl'
mutable struct Dense{T<:AbstractFloat, M<:AbstractMatrix{T}, VN<:Union{AbstractVector{T}, Nothing}} <: Layer
    w::M # weight
    b::VN # bias
end

function Dense(init_w, x_size, size, use_bias, precision = Float32)
    w = convert.(precision, rand(init_w(x_size), size, x_size))
    b = use_bias ? zeros(precision, size) : nothing
    return Dense(w, b)
end

abstract type Model end

# functor, see 'core.jl'
struct NeuralNetwork{VT<:AbstractVector{Layer}} <: Model
    layers::VT
end

function NeuralNetwork(x_size, w_inits, sizes, use_biases, precision = Float32)
    layers = Vector{Layer}(undef, 0)
    params = zip(w_inits, pushfirst!(sizes[begin:end - 1], x_size), sizes, use_biases)

    for (init_w, x_size, size, use_bias) in params
        # TODO: parameterize layer type
        layer = Dense(init_w, x_size, size, use_bias, precision)
        push!(layers, layer)
    end

    return NeuralNetwork(layers)
end

struct Linear{T<:AbstractFloat, S<:Union{AbstractVector{T}, T}, R<:Union{T, Nothing}} <: Model
    w::S
    b::R
end

# not implemented yet
struct GenerativeAdversarialNetwork{T1<:Model, T2<:Model} <: Model
    generator::T1
    discriminator::T2
end
