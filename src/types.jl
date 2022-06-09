
struct Data{A1<:AbstractArray, A2<:AbstractArray}
    input::A1
    label::A2
end

# corresponds to a layer in a 'Neural_Network'
struct Layer_Parameter{F1<:Function, F2<:Function, F3<:Function, T<:AbstractFloat}
    norm_func::F1
    activ_func::F2
    regular_func::F3
    regular_rate::T
    learn_rate::T
end

# functor, see 'core.jl'
struct Epoch_Parameter{T<:Integer, F1<:Function, F2<:Function, H<:Layer_Parameter}
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
struct Neural_Network{T<:Layer} <: Model
    layers::Vector{T}
end

function Neural_Network(input_size, precision, weight_init_funcs, sizes, use_biases)
    tmp_sizes = input_size, sizes...

    # TODO: improve readability
    weights = [convert(Matrix{precision},
        rand(weight_init_func(input_size), output_size, input_size))
            for (weight_init_func, input_size, output_size) in zip(weight_init_funcs, tmp_sizes[begin:end - 1], tmp_sizes[begin + 1:end])]
    biases = [use_bias ? zeros(precision, size) : nothing for (use_bias, size) in zip(use_biases, sizes)]

    # TODO: remove splatting
    # TODO: parameterize layer type
    layers = map(args -> Dense(args...), zip(weights, biases))
    return Neural_Network(layers)
end

# not implemented yet
struct GAN{T1<:Model, T2<:Model} <: Model
    generator::T1
    discriminator::T2
end
