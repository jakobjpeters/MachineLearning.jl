
# calculate and cache 'Z' and 'output'
function (dense::Dense)(input, activ_func, cache)
    cache.Z = dense.weight * input
    if !isnothing(dense.bias)
        cache.Z .+= dense.bias
    end

    # reduce allocations by enabling 'map!'
    cache.output = preallocate(cache.output, cache.Z)
    map!(activ_func, cache.output, cache.Z) # |> layer.norm_func

    return Nothing
end

# propagate input -> output through each layer
function (neural_net::Neural_Network)(input, layer_params, caches)
    for i in 1:length(neural_net.layers)
        layer_input = i == 1 ? input : caches[i - 1].output
        neural_net.layers[i](layer_input, layer_params[i].activ_func, caches[i])
    end

    return nothing
end

# regular_func::typeof(T) where T <: Uniont{weight_decay, l1, l2} when l2 doesn't use an adaptive gradient
function update_weight!(regular_func, regular_rate, learn_rate, weight, δl_δz, layer_input, batch_size)
    # gemm!(tA, tB, α, A, B, β, C) = α * A * B + β * C -> C where 'T' indicates a transpose
    gemm!('N', 'T', -learn_rate / batch_size, δl_δz, layer_input, one(eltype(layer_input)) - regular_rate, weight)
end

function update_weight!(regular_func::typeof(l1), regular_rate, learn_rate, weight, δl_δz, layer_input, batch_size)
    gradient = δl_δz * transpose(layer_input) / batch_size
    penalty = derivative(regular_func).(weight, regular_rate / learn_rate)
    # axpy!(α, X, Y) = α * X + Y -> Y
    axpy!(-learn_rate, gradient + penalty, weight)
end

# needed for 'l2' with adaptive gradient such as ADAM, although ADAM with weight decay is better
# https://arxiv.org/pdf/1711.05101v3.pdf
# function update_weight!(optimiser::T, regular_func::typeof(l2), regular_rate, learn_rate, weight, δl_δz, layer_input, batch_size)
#     gradient = δl_δz * transpose(layer_input) / batch_size
#     penalty = derivative(regular_func)(weight, regular_rate / learn_rate)
#     # axpy!(α, X, Y) = α * X + Y -> Y
#     axpy!(-learn_rate, gradient + penalty, weight)
# end

# calculate the gradient and update the model's parameters for a batched input
@inline function backpropagate!(layers, cost_func, layer_params, caches, input, label)
    n_layers = length(layers)
    batch_size = size(input, 2)
    caches[n_layers].δl_δa = derivative(cost_func)(label, caches[end].output)

    for i in reverse(1:n_layers)
        # calculate intermediate partial derivatives
        δa_δz = map(derivative(layer_params[i].activ_func), caches[i].Z)
        # reduce allocations by enabling '.='
        caches[i].δl_δz = preallocate(caches[i].δl_δz, caches[i].Z)
        caches[i].δl_δz .= caches[i].δl_δa .* δa_δz

        # do not need to calculate δl_δa for first layer, since there are no parameters to update
        if i != 1
            caches[i - 1].δl_δa = transpose(layers[i].weight) * caches[i].δl_δz
        end

        layer_input = i == 1 ? input : caches[i - 1].output

        # update weights and biases in-place
        # update_weight!(layer_params[i].regular_func, layer_params[i].learn_rate, batch_size, caches[i].δl_δz, layer_input, layer_params[i].regular_rate, layers[i].weight)
        update_weight!(layer_params[i].regular_func, layer_params[i].regular_rate, layer_params[i].learn_rate, layers[i].weight, caches[i].δl_δz, layer_input, batch_size)
        if layers[i].bias !== nothing
            δl_δb = dropdims(sum(caches[i].δl_δz, dims = 2), dims = 2)
            # axpy!(α, X, Y) = α * X + Y -> Y
            axpy!(-layer_params[i].learn_rate / batch_size, δl_δb, layers[i].bias)
        end
    end

    return nothing
end

# test the model and return its accuracy and loss for each data split
function assess!(dataset, model, cost_func, layer_params, caches)
    precision = eltype(model.layers[begin].weight)
    accuracies = Vector{precision}()
    costs = Vector{precision}()

    for split in dataset
        model(split.input, layer_params, caches)
        n = size(caches[end].output, 2)

        # TODO: parameterize decision criterion
        criterion = pair -> argmax(pair[begin]) == argmax(pair[end])
        n_correct = count(criterion, zip(eachcol(caches[end].output), eachcol(split.label)))
        push!(accuracies, n_correct / n)

        penalty = sum([sum(layer_params[i].regular_func.(model.layers[i].weight, layer_params[i].regular_rate)) for i in 1:length(layer_params)])
        cost = sum(cost_func(split.label, caches[end].output))
        push!(costs, (cost + penalty) / n)
    end

    return accuracies, costs
end

# coordinate an epoch of model training
function (epoch_param::Epoch_Parameter)(model, layer_params, caches, input, label)

    if epoch_param.shuffle && epoch_param.batch_size < size(input, 1)
        input, label = shuffle_pair(input, label)
    end

    # train model for each batch
    for first in 1:epoch_param.batch_size:size(input, 2)
        last = min(size(input, 2), first + epoch_param.batch_size - 1)
        norm_input = epoch_param.norm_func(view(input, :, first:last))

        model(norm_input, layer_params, caches)
        backpropagate!(model.layers, epoch_param.cost_func, layer_params, caches, norm_input, view(label, :, first:last))
    end

    return nothing
end
