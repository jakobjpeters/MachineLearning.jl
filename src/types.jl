
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

# functor, see 'functors.jl'
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
struct LayerParameters{F, R<:Regularizer}
    η::Float32 # learning rate
    normalize::F
    regularizer::R
end

function LayerParameters(η, normalize, regularize)
    return LayerParameters(convert(Float32, η), normalize, regularize)
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

# functor, see 'functors.jl'
mutable struct Dense{F, M<:AbstractMatrix{Float32}, VN<:Union{AbstractVector{Float32}, Nothing}} <: Layer
    activate::F
    w::M # weight
    b::VN # bias
end

function Dense(x_size, y_size, activate, init_w, use_bias)
    w = convert.(Float32, rand(init_w(x_size), y_size, x_size))
    b = use_bias ? zeros(Float32, y_size) : nothing

    return Dense(activate, w, b)
end

abstract type Model end

# functor, see 'functors.jl'
struct NeuralNetwork{F, VL<:AbstractVector{<:Layer}} <: Model
    loss::F
    layers::VL
end

function NeuralNetwork(loss, x_size, y_sizes::AbstractArray, activators::AbstractArray, inits_w::AbstractArray, use_biases::AbstractArray)
    x_sizes = pushfirst!(y_sizes[begin:end - 1], x_size)
    layers_params = zip(x_sizes, y_sizes, activators, inits_w, use_biases)
    layers = map(layer_params -> Dense(layer_params...), layers_params)

    return NeuralNetwork(loss, layers)
end

function NeuralNetwork(loss, x_size, y_sizes, activate, w_init = xavier, use_bias = true)
    params = [x_size, y_sizes]
    for param in [activate, w_init, use_bias]
        push!(params, isa(param, AbstractArray) ? param : repeat([param], length(y_sizes)))
    end

    return NeuralNetwork(loss, params...)
end

# functor, see 'functors.jl'
mutable struct Linear{F, S<:Union{AbstractVector{Float32}, Float32}, R<:Union{Float32, Nothing}} <: Model
    loss::F
    w::S
    b::R
end

function Linear(loss, w, b = nothing)
    w = convert.(Float32, w)
    b = isnothing(b) ? nothing : convert(Float32, b)
    return Linear(loss, w, b)
end

function Linear(loss, use_bias::Bool = true)
    b = use_bias ? 0.0 : nothing
    return Linear(loss, 0.0, b)
end

# not implemented yet
struct GenerativeAdversarialNetwork{T1<:Model, T2<:Model} <: Model
    generator::T1
    discriminator::T2
end
