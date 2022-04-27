
function assess!(model, inputs, labels)
    correct = 0
    error = 0.0

    for (input, label) in zip(inputs, labels)
        model(input)

        if issetequal(findall(x -> x > 0.5, model.activations[end]), label)
            correct += 1
        end

        error += mean(model.cost_func(model.activations[end], label))
    end

    return correct / length(labels), error / length(labels)
end

function backpropagate!(model, inputs, labels)
    for (input, label) in zip(inputs, labels)
        model(input)

        δl_δa = deriv(model.cost_func, model.activations[end], label)

        for (layer, activations) in zip(reverse(model.layers), reverse(model.activations[begin:end - 1]))
            δl_δb = δl_δa .* deriv(layer.activ_func, layer.Zs)

            layer.δl_δw -= δl_δb * transpose(activations)
            if !isnothing(layer.biases)
                layer.δl_δb -= δl_δb
            end

            if layer === model.layers[begin]
                break
            end

            δl_δa = transpose(layer.weights) * δl_δb
        end
    end

    for (i, layer) in enumerate(model.layers)
        layer.weights += model.learn_rates[i] * layer.δl_δw
        fill!(model.layers[i].δl_δw, 0.0)

        if !isnothing(layer.biases)
            layer.biases += model.learn_rates[i] * layer.δl_δb
            fill!(layer.δl_δb, 0.0)
        end
    end
end
