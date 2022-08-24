
using LinearAlgebra: BLAS.gemm!, axpy!

# calculate and cache linear and activation
function predict!(layer::Dense, x, cache)
    cache.l = linear(layer.w, x, layer.b)

    cache.a = preallocate(cache.a, cache.l)
    map!(layer.activate, cache.a, cache.l) # |> layer.norm_func

    return cache.a
end

# propagate input -> linear -> activation through each layer
function predict!(model::NeuralNetwork, x, caches)
    for (layer, cache) in zip(model.layers, caches)
        x = predict!(layer, x, cache)
    end

    return caches[end].a
end

# regularize::typeof(T) where T <: Uniont{weight_decay, l1, l2} when 'l2' doesn't use an adaptive gradient
function update_weight!(regularize, λ, η, w, δe_δl, x)
    # gemm!(tA, tB, α, A, B, β, C) = α * A * B + β * C -> C where 'T' indicates a transpose
    gemm!('N', 'T', -η / size(x, 2), δe_δl, x, one(eltype(x)) - λ, w)
    
    return w
end

# slower, only call if 'λ != 0'
# function update_weight!(regularize::typeof(l1), λ, η, w, δe_δl, x)
#     ∇ = zeros(eltype(x))
#     ∇ = δe_δl * transpose(x) / size(x, 2)
#     penalty = derivative(regularize)(w, λ / η)
#     # axpy!(α, X, Y) = α * X + Y -> Y
#     axpy!(-η, ∇ + penalty, w)

#     return w
# end

# needed for 'l2' with adaptive gradient such as ADAM, although ADAM with weight decay is better
# https://arxiv.org/pdf/1711.05101v3.pdf
# function update_weight!(optimiser::T, regular_func::typeof(l2), λ, α, w, δe_δl, x)
#     ∇ = δe_δl * transpose(x) / size(x, 2)
#     penalty = derivative(regular_func)(w, λ / α)
#     # axpy!(α, X, Y) = α * X + Y -> Y
#     axpy!(-α, ∇ + penalty, w)
# end


@inline function update_params!(regularizer, η, w, δe_δl, x, b)
    update_weight!(regularizer.regularize, regularizer.λ, η, w, δe_δl, x)

    if !isnothing(b)
        δe_δb = dropdims(sum(δe_δl, dims = 2), dims = 2)
        # axpy!(α, X, Y) = α * X + Y -> Y
        axpy!(-η / size(x, 2), δe_δb, b)
    end

    return nothing
end

# calculate the gradient and update the model's parameters for a batched input
@inline function backpropagate!(model, layers_params, caches, dataset)
    predict!(model, dataset.x, caches)

    δe_δa = derivative(model.loss)(dataset.y, caches[end].a)

    for i in reverse(eachindex(model.layers))
        layer_x = i == 1 ? dataset.x : caches[i - 1].a
        
        # calculate intermediate partial derivatives
        δa_δl = map(derivative(model.layers[i].activate), caches[i].l)

        caches[i].δe_δl = preallocate(caches[i].δe_δl, caches[i].l)
        caches[i].δe_δl .= δe_δa .* δa_δl

        # do not need to calculate δe_δa for first layer, since there are no parameters to update
        δe_δa = i == 1 ? nothing : transpose(model.layers[i].w) * caches[i].δe_δl

        update_params!(
            layers_params[i].regularizer,
            layers_params[i].η,
            model.layers[i].w,
            caches[i].δe_δl,
            layer_x,
            model.layers[i].b
        )
    end

    return model
end

# test the model and return its accuracy and cost for each data split
function assess(datasets, model, regularizers)
    accuracies = Float32[]
    costs = Float32[]

    # TODO: parameterize decision criterion
    criterion = pair -> argmax(pair[begin]) == argmax(pair[end])

    for dataset in datasets
        ŷ = model(dataset.x)
        n = size(dataset.x, 2)

        n_correct = count(criterion, zip(eachcol(dataset.y), eachcol(ŷ)))
        push!(accuracies, n_correct / n)

        cost = sum(model.loss(dataset.y, ŷ))

        penalty = zero(Float32)
        for (layer, regularizer) in zip(model.layers, regularizers)
            penalty += sum(regularizer(layer.w))
        end
        
        push!(costs, (cost + penalty) / n)
    end

    return @NamedTuple{accuracies::Vector{Float32}, costs::Vector{Float32}}((accuracies, costs))
end

function assess(datasets, model)
    costs = Vector{Float32}(undef, 0)

    for dataset in datasets
        ŷ = model(dataset.x)

        cost = mean(model.loss(dataset.y, ŷ))
        push!(costs, cost)
    end

    return @NamedTuple{costs::Vector{Float32}}((costs,))
end

# coordinate an epoch of model training
function train!(model, dataset, batch_size, layers_params, caches, normalize = z_score, shuffle_data = true)
    if shuffle_data && batch_size < dataset.n # TODO: && shuffle!(x, y)
        dataset = shuffle(dataset)
    end

    # train model for each batch
    for first in 1:batch_size:dataset.n
        last = min(dataset.n, first + batch_size - 1)
        x = normalize(view(dataset.x, :, first:last))
        y = view(dataset.y, :, first:last)
        batch = Dataset(x, y)

        backpropagate!(model, layers_params, caches, batch)
    end

    return model
end

function train!(model::Linear{<:AbstractVector}, x, y)
    throw(ErrorException("Multiple regression not implemented yet"))

    return model
end

function train!(model::Linear, dataset)
    x̄ = mean(dataset.x)
    ȳ = mean(dataset.y)

    model.w = sum((dataset.x .- x̄) .* (dataset.y .- ȳ)) / sum((dataset.x .- x̄) .^ 2)
    if !isnothing(model.b)
        model.b = ȳ - x̄ * model.w
    end

    return model
end