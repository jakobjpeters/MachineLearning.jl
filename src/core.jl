
# calculate and cache 'linear' and 'activation'
function (dense::Dense)(x, activate, cache)
    cache.l = dense.w * x
    if !isnothing(dense.b)
        cache.l .+= dense.b
    end

    # reduce allocations by enabling 'map!'
    cache.a = preallocate(cache.a, cache.l)
    map!(activate, cache.a, cache.l) # |> layer.norm_func

    return
end

# propagate input -> linear -> activation through each layer
function (neural_net::NeuralNetwork)(x, layers_params, caches)
    for i in eachindex(neural_net.layers)
        layer_x = i == 1 ? x : caches[i - 1].a
        neural_net.layers[i](layer_x, layers_params[i].activate, caches[i])
    end

    return
end

# regular_func::typeof(T) where T <: Uniont{weight_decay, l1, l2} when l2 doesn't use an adaptive gradient
function update_weight!(regularize, λ, η, w, δe_δl, x)
    # gemm!(tA, tB, α, A, B, β, C) = α * A * B + β * C -> C where 'T' indicates a transpose
    gemm!('N', 'T', -η / size(x, 2), δe_δl, x, one(eltype(x)) - λ, w)
    
    return
end

function update_weight!(regularize::typeof(l1), λ, η, w, δe_δl, x)
    ∇ = δe_δl * transpose(x) / size(x, 2)
    penalty = derivative(regularize).(w, λ / η)
    # axpy!(α, X, Y) = α * X + Y -> Y
    axpy!(-η, ∇ + penalty, w)

    return
end

# needed for 'l2' with adaptive gradient such as ADAM, although ADAM with weight decay is better
# https://arxiv.org/pdf/1711.05101v3.pdf
# function update_weight!(optimiser::T, regular_func::typeof(l2), λ, α, w, δe_δl, x)
#     ∇ = δe_δl * transpose(x) / size(x, 2)
#     penalty = derivative(regular_func)(w, λ / α)
#     # axpy!(α, X, Y) = α * X + Y -> Y
#     axpy!(-α, ∇ + penalty, w)
# end

function train_layer!(i, x, layer, layer_params, cache)
    # calculate intermediate partial derivatives
    δa_δl = map(derivative(layer_params.activate), cache.l)
    # reduce allocations by enabling '.='
    cache.δe_δl = preallocate(cache.δe_δl, cache.l)
    cache.δe_δl .= cache.δe_δa .* δa_δl

    # do not need to calculate δl_δa for first layer, since there are no parameters to update
    δe_δa = i == 1 ? nothing : transpose(layer.w) * cache.δe_δl

    # update weights and biases in-place
    update_weight!(layer_params.regularize, layer_params.λ, layer_params.η, layer.w, cache.δe_δl, x)
    if layer.b !== nothing
        δe_δb = dropdims(sum(cache.δe_δl, dims = 2), dims = 2)
        # axpy!(α, X, Y) = α * X + Y -> Y
        axpy!(-layer_params.η / size(x, 2), δe_δb, layer.b)
    end

    return δe_δa
end

# calculate the gradient and update the model's parameters for a batched input
@inline function backpropagate!(layers, layers_params, caches, x, y, loss)
    caches[length(layers)].δe_δa = derivative(loss)(y, caches[end].a)

    for i in reverse(eachindex(layers))
        layer_x = i == 1 ? x : caches[i - 1].a
        δe_δa = train_layer!(i, layer_x, layers[i], layers_params[i], caches[i])

        i != 1 || break
        caches[i - 1].δe_δa = δe_δa
    end

    return
end

# test the model and return its accuracy and cost for each data split
function assess!(dataset, model, loss, layers_params, caches)
    precision = eltype(model.layers[begin].w)
    accuracies = Vector{precision}()
    costs = Vector{precision}()

    for split in dataset
        model(split.x, layers_params, caches)
        n = size(caches[end].a, 2)

        # TODO: parameterize decision criterion
        criterion = pair -> argmax(pair[begin]) == argmax(pair[end])
        n_correct = count(criterion, zip(eachcol(caches[end].a), eachcol(split.y)))
        push!(accuracies, n_correct / n)

        penalty = sum([sum(layers_params[i].regularize.(model.layers[i].w, layers_params[i].λ)) for i in eachindex(layers_params)])
        cost = sum(loss(split.y, caches[end].a))
        push!(costs, (cost + penalty) / n)
    end

    return accuracies, costs
end

# coordinate an epoch of model training
function (epoch::Epoch)(model, layers_params, caches, x, y)
    if epoch.shuffle && epoch.batch_size < size(x, 1)
        x, label = shuffle_pair(x, y)
    end

    # train model for each batch
    for first in 1:epoch.batch_size:size(x, 2)
        last = min(size(x, 2), first + epoch.batch_size - 1)
        norm_x = epoch.normalize(view(x, :, first:last))

        model(norm_x, layers_params, caches)
        backpropagate!(model.layers, layers_params, caches, norm_x, view(label, :, first:last), epoch.loss)
    end

    return
end

function train_model!(epoch, model, caches, dataset, assessments, display = nothing, n_epochs = 1)
    for i in 1:n_epochs
        @time epoch(model, epoch.layers_params, caches, dataset[begin].x, dataset[begin].y)
        @time push!(assessments, Assessment(assess!(dataset, model, epoch.loss, epoch.layers_params, caches)))

        # see 'interface.jl'
        display(assessments)
    end

    return
end

function train_model!(epochs::Vector, model, caches, dataset, assessments, display = nothing)
    for epoch in epochs
        train_model!(epoch, model, caches, dataset, assessments, display)
    end

    return
end

