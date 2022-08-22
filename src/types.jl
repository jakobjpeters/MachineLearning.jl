
struct Dataset{A1<:AbstractArray{Float32}, A2<:AbstractArray{Float32}}
    x::A1
    y::A2
    n::Int64
end

# TODO: make internal constructor
function Dataset(x::AbstractArray{Float32, N}, y::AbstractArray{Float32, N}) where N
    return Dataset(x, y, size(x, N))
end

function Dataset(x::AbstractArray{<:Number, N}, y::AbstractArray{<:Number, N}) where N
    return Dataset(convert.(Float32, x), convert.(Float32, y))
end

struct Regularizer{F}
    regularize::F
    λ::Float32
end

function Regularizer(regularize, λ)
    return Regularizer(regularize, convert(Float32, λ))
end

function Regularizer()
    return Regularizer(weight_decay, 0.0)
end

# corresponds to layers in a 'Neural_Network'
struct LayerParameters{F1, F2, R<:Regularizer}
    η::Float32 # learning rate
    activate::F1
    normalize::F2
    regularizer::R
end

function LayerParameters(η, params...)
    return LayerParameters(convert(Float32, η), params...)
end

# corresponds to a layer in a 'Neural_Network'
mutable struct Cache{M<:AbstractMatrix{Float32}}
    δe_δl::M # δ error / δ linear
    l::M # linear
    a::M # activation
end

# TODO: make internal constructor
function Cache()
    init = Float32[;;]
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

function NeuralNetwork(x_size, sizes::AbstractArray, w_inits::AbstractArray, use_biases::AbstractArray)
    x_sizes = pushfirst!(sizes[begin:end - 1], x_size)
    layers_params = zip(w_inits, x_sizes, sizes, use_biases)
    layers = map(layer_params -> Dense(layer_params...), layers_params)

    return NeuralNetwork(layers)
end

function NeuralNetwork(x_size, sizes, w_init = xavier, use_bias = true)
    params = [x_size, sizes]
    for param in [w_init, use_bias]
        push!(params, isa(param, AbstractArray) ? param : repeat([param], length(sizes)))
    end

    return NeuralNetwork(params...)
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
