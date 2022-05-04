
# 'Layer' functor
# given input, calculate and cache 'Zs' and 'activations'
function (layer::Layer)(activ_func, cache, input)
    cache.Zs = layer.weights * input
    if !isnothing(layer.biases)
        cache.Zs .+= layer.biases
    end

    cache.activations = cache.Zs |> activ_func # |> layer.norm_func

    # should 'return nothing'
    return cache.activations
end

# 'Neural_Network' functor
# call each layer with its correct parameters
function (neural_net::Neural_Network)(input, h_params, caches)
    for (layer, cache, h_param) in zip(neural_net.layers, caches, h_params)
        # fix: allocating
        input = layer(h_param.activ_func, cache, input)
    end

    return nothing
end

# given a model, calculate and cache the gradient for a batch of inputs
function backpropagate!(model, cost_func, h_params, caches, inputs, labels)
    model(inputs, h_params, caches)

    # fix: allocating
    δl_δa = deriv(cost_func, caches[end].activations, labels)
    prev_activations = pushfirst!(map(cache -> cache.activations, caches[begin:end - 1]), inputs)
    activ_funcs = map(h_param -> h_param.activ_func, h_params)
    
    # iterate end to begin to calculate each layer's gradient
    for (layer, cache, activ_func, prev_activation) in zip(reverse(model.layers), reverse(caches), reverse(activ_funcs), reverse(prev_activations))
        δl_δz = δl_δa .* deriv(activ_func, cache.Zs)

        # cache gradients
        # gradients are averaged (divided by 'batch_size') in 'apply_gradient!' for efficiency
        cache.δl_δw = δl_δz * transpose(prev_activation)
        if layer.biases !== nothing
            cache.δl_δb = dropdims(sum(δl_δz, dims = 2), dims = 2)
        end

        # if first layer, this calculation is not needed
        layer === model.layers[begin] && break
        δl_δa = transpose(layer.weights) * δl_δz
    end

    return nothing
end

# update weights and biases with cached gradient
function apply_gradient!(layers, learn_rates, caches, batch_size)
    # dividing by 'batch_size' turns the gradients from a sum to an average
    scales = learn_rates / batch_size

    # update each layer's weights and biases, then reset its cache
    for (layer, cache, scale) in zip(layers, caches, scales)
        layer.weights += cache.δl_δw * scale 
        if layer.biases !== nothing
            layer.biases += cache.δl_δb * scale
        end
    end

    return nothing
end

# given a model and data, test the model and return its accuracy and loss
function assess!(model, cost_func, h_params, caches, inputs, labels)
    model(inputs, h_params, caches)

    # TODO: parameterize decision criteria
    criteria = z -> argmax(first(z)) == argmax(last(z))
    accuracy = count(criteria, zip(eachcol(caches[end].activations), eachcol(labels))) / size(inputs, 2)

    loss = z -> mean(cost_func(z[1], z[2]))
    cost = mean(map(loss, zip(eachcol(caches[end].activations), eachcol(labels))))

    return accuracy, cost
end

# 'Epoch' functor
# given a model and data, coordinate model training
function (epoch::Epoch)(model, h_params, caches, inputs, labels)

    if epoch.shuffle && epoch.batch_size < size(inputs, 1)
        inputs, labels = shuffle_pair(inputs, labels)
    end

    # train model for each batch
    for first in 1:epoch.batch_size:size(inputs, 2)
        last = min(size(inputs, 2), first + epoch.batch_size - 1)
        backpropagate!(model, epoch.cost_func, h_params, caches, view(inputs, :, first:last), view(labels, :, first:last))
        apply_gradient!(model.layers, map(h_params -> h_params.learn_rate, h_params), caches, epoch.batch_size)
    end

    return nothing
end
