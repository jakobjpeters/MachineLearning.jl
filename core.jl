
function assess!(model, inputs, labels)
    correct = 0
    error = 0.0

    for (input, label) in zip(inputs, labels)
        model(input)

        if argmax(model.layers[end].activations) == label[1]
            correct += 1
        end

        error += mean(model.cost_func(model.layers[end].activations, label))
    end

    return correct / length(labels), error / length(labels)
end

function backpropagate!(model, inputs, labels)
    for (input, label) in zip(inputs, labels)
        model(input)
        activations = deepcopy([layer.activations for layer in model.layers[begin:end - 1]])
        prev_activations = push!(reverse(activations), input)

        δl_δa = deriv(model.cost_func, model.layers[end].activations, label)

        for (layer, prev_activation) in zip(reverse(model.layers), prev_activations)
            δl_δb = δl_δa .* deriv(layer.activ_func, layer.Zs)

            layer.δl_δw -= δl_δb * transpose(prev_activation)
            if !isnothing(layer.biases)
                layer.δl_δb -= δl_δb
            end

            if layer === model.layers[begin]
                break
            end

            δl_δa = transpose(layer.weights) * δl_δb
        end
    end

    for layer in model.layers
        layer.weights += layer.learn_rate * layer.δl_δw
        fill!(layer.δl_δw, 0.0)

        if !isnothing(layer.biases)
            layer.biases += layer.learn_rate * layer.δl_δb
            fill!(layer.δl_δb, 0.0)
        end
    end
end
