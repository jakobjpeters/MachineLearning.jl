
struct Data{A1<:AbstractArray, A2<:AbstractArray}
    x::A1
    y::A2
end

# corresponds to a layer in a 'Neural_Network'
struct LayerParameters{F1<:Function, F2<:Function, F3<:Function, T<:AbstractFloat}
    normalize::F1
    activate::F2
    regularize::F3
    λ::T # regularizer rate
    η::T # learning rate
end

# functor, see 'core.jl'
struct Epoch{T<:Integer, F1<:Function, F2<:Function, H<:LayerParameters}
    batch_size::T
    shuffle::Bool
    loss::F1
    normalize::F2
    layers_params::Vector{H}
end

# corresponds to a layer in a 'Neural_Network'
mutable struct Cache{M<:AbstractMatrix}
    δe_δl::M # δ error / δ linear
    δe_δa::M # δ error / δ activation
    l::M # linear
    a::M # activation
end

function Cache(precision)
    return Cache(repeat([Matrix{precision}(undef, 0, 0)], length(fieldnames(Cache)))...)
end

abstract type Layer end

# functor, see 'core.jl'
mutable struct Dense{M<:AbstractMatrix, VN<:Union{AbstractVector, Nothing}} <: Layer
    w::M # weight
    b::VN # bias
end

abstract type Model end

# functor, see 'core.jl'
struct NeuralNetwork{T<:Layer} <: Model
    layers::Vector{T}
end

function NeuralNetwork(x_size, precision, w_inits, sizes, use_biases)
    layers = Vector{Layer}(undef, 0)
    params = zip(w_inits, pushfirst!(sizes[begin:end - 1], x_size), sizes, use_biases)

    for (init_w, x_size, size, use_bias) in params
        w = convert.(precision, rand(init_w(x_size), size, x_size))
        b = use_bias ? zeros(precision, size) : nothing

        push!(layers, Dense(w, b))
    end

    return NeuralNetwork(layers)
end

# not implemented yet
struct GenerativeAdversarialNetwork{T1<:Model, T2<:Model} <: Model
    generator::T1
    discriminator::T2
end
