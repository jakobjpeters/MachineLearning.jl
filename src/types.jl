
struct Data{A1<:AbstractArray, A2<:AbstractArray}
    input::A1
    label::A2
end

# corresponds to a layer in a 'Neural_Network'
struct LayerParameter{F1<:Function, F2<:Function, F3<:Function, T<:AbstractFloat}
    norm_func::F1
    activ_func::F2
    regular_func::F3
    regular_rate::T
    learn_rate::T
end

# functor, see 'core.jl'
struct EpochParameter{T<:Integer, F1<:Function, F2<:Function, H<:LayerParameter}
    batch_size::T
    shuffle::Bool
    cost_func::F1
    norm_func::F2
    layer_param::Vector{H}
end

# corresponds to a layer in a 'Neural_Network'
mutable struct Cache{M<:AbstractMatrix}
    δl_δz::M
    δl_δa::M
    output::M
    Z::M
end

function Cache(precision)
    return Cache(repeat([Matrix{precision}(undef, 0, 0)], length(fieldnames(Cache)))...)
end

abstract type Layer end

# functor, see 'core.jl'
mutable struct Dense{M<:AbstractMatrix, VN<:Union{AbstractVector, Nothing}} <: Layer
    weight::M
    bias::VN
end

abstract type Model end

# functor, see 'core.jl'
struct NeuralNetwork{T<:Layer} <: Model
    layers::Vector{T}
end

function NeuralNetwork(input_size, precision, weight_init_funcs, sizes, use_biases)
    layers = Vector{Layer}(undef, 0)

    for (weight_init_func, input_size, output_size, use_bias) in zip(weight_init_funcs, pushfirst!(sizes[begin:end - 1], input_size), sizes, use_biases)
        weight = convert.(precision, rand(weight_init_func(input_size), output_size, input_size))
        bias = use_bias ? zeros(precision, output_size) : nothing
        push!(layers, Dense(weight, bias))
    end

    return NeuralNetwork(layers)
end

# not implemented yet
struct GenerativeAdversarialNetwork{T1<:Model, T2<:Model} <: Model
    generator::T1
    discriminator::T2
end
