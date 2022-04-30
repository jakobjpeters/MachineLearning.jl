
function (layer::Layer)(h_params, cache, input)
    cache.Zs = layer.weights * input
    if !isnothing(layer.biases)
        cache.Zs += layer.biases
    end

    cache.activations = cache.Zs |> h_params.activ_func # |> layer.norm_func

    # should 'return nothing'
    return cache.activations
end

function (neural_net::Neural_Network)(input)
    for (layer, cache, h_param) in zip(neural_net.layers, neural_net.caches, neural_net.h_params)
        # fix: allocating
        input = layer(h_param, cache, input)
    end

    return nothing
end

function backpropagate!(model, inputs, labels)
    for (input, label) in zip(inputs, labels)
        model(input)

        δl_δa = deriv(model.cost_func, model.caches[end].activations, label)
        prev_activations = input, map(cache -> cache.activations, model.caches[begin:end - 1])...
        activ_funcs = map(h_param -> h_param.activ_func, model.h_params)
        
        # fields = zip(map(field -> reverse(field), [model.layers, model.caches, activ_funcs, prev_activations]))
        for (layer, cache, activ_func, prev_activation) in zip(reverse(model.layers), reverse(model.caches), reverse(activ_funcs), reverse(prev_activations))
            δl_δb = δl_δa .* deriv(activ_func, cache.Zs)

            cache.δl_δw -= δl_δb * transpose(prev_activation)
            if layer.biases !== nothing
                cache.δl_δb -= δl_δb
            end

            layer === model.layers[begin] && break

            δl_δa = transpose(layer.weights) * δl_δb
        end
    end

    return nothing
end

function apply_gradient!(layers, caches, learn_rates, batch_size)
    scales = learn_rates / batch_size

    for (layer, cache, scale) in zip(layers, caches, scales)
        layer.weights += cache.δl_δw * scale 
        fill!(cache.δl_δw, 0.0)
        if layer.biases !== nothing
            layer.biases += cache.δl_δb * scale
            fill!(cache.δl_δb, 0.0)
        end
    end

    return nothing
end

function assess!(model, inputs, labels)
    correct = 0
    error = 0.0

    for (input, label) in zip(inputs, labels)
        model(input)

        if argmax(model.caches[end].activations) == label[1]
            correct += 1
        end

        error += mean(model.cost_func(model.caches[end].activations, label))
    end

    return correct / length(labels), error / length(labels)
end

function (epoch::Epoch)(model, inputs, labels)
    if epoch.shuffle && epoch.batch_size < length(inputs)
        inputs, labels = shuffle_data(inputs, labels)
    end

    for first in 1:epoch.batch_size:length(inputs)
        last = min(length(inputs), first + epoch.batch_size - 1)
        backpropagate!(model, view(inputs, first:last), view(labels, first:last))
        apply_gradient!(model.layers, model.caches, map(h_params -> h_params.learn_rate, model.h_params), epoch.batch_size)
    end

    return nothing
end
