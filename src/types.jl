
struct Data{A1<:AbstractArray, A2<:AbstractArray}
    inputs::A1
    labels::A2
end

# functor, see 'core.jl'
struct Epoch{T<:Integer, F<:Function}
    batch_size::T
    shuffle::Bool
    cost_func::F
end

# corresponds to a layer in a 'Neural_Network'
mutable struct Cache{M<:AbstractMatrix, VN<:Union{AbstractVector, Nothing}}
    δl_δw::M
    δl_δb::VN
    activations::M
    Zs::M
end

# given a 'Neural_Network', construct a list of 'Cache's to prevent redundant calculations in 'core.jl'
function make_caches(neural_net)
    sizes = map(layer -> size(layer.weights, 1), neural_net.layers)

    δl_δw = map(layer -> fill!(deepcopy(layer.weights), 0.0), neural_net.layers)
    δl_δb = map(layer -> deepcopy(layer.biases), neural_net.layers)
    activations = map(i -> Matrix{Float64}(undef, 0, 0), sizes)
    Zs = deepcopy(activations)
    
    # TODO: remove splatting
    return map(args -> Cache(args...), zip(δl_δw, δl_δb, activations, Zs))
end

# corresponds to a layer in a 'Neural_Network'
struct Hyperparameters{F1<:Function, F2<:Function, T<:AbstractFloat}
    norm_func::F1
    activ_func::F2
    learn_rate::T
end

abstract type Layer end

# functor, see 'core.jl'
mutable struct Dense{M<:AbstractMatrix, VN<:Union{AbstractVector, Nothing}} <: Layer
    weights::M
    biases::VN
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