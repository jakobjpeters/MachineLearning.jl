
function predict!(nn, input)
    nn.activations[begin] = input

    for layer_n in 1:length(nn.sizes)
        nn.activations[layer_n] = nn.norm_funcs[layer_n](nn.activations[layer_n])

        nn.Zs[layer_n] = nn.weights[layer_n] * nn.activations[layer_n]
        if nn.use_biases[layer_n]
            nn.Zs[layer_n] += nn.biases[layer_n]

        nn.activations[layer_n + 1] = nn.activ_funcs[layer_n](nn.Zs[layer_n])
        end
    end
end

function assess!(nn, inputs, labels)
    correct = 0
    error = 0.0

    for (input, label) in zip(inputs, labels)
        predict!(nn, input)

        if issetequal(findall(x -> x > 0.5, nn.activations[end]), label)
            correct += 1
        end

        error += mean(nn.cost_func(nn.activations[end], label))
    end

    return correct / length(labels), error / length(labels)
end

function backpropagate!(nn, inputs, labels)
    for (input, label) in zip(inputs, labels)
        predict!(nn, input)

        δl_δa = deriv(nn.cost_func, nn.activations[end], label)

        for layer_n in reverse(1:length(nn.sizes))
            δl_δb = δl_δa .* deriv(nn.activ_funcs[layer_n], nn.Zs[layer_n])

            nn.δl_δw[layer_n] -= δl_δb * transpose(nn.activations[layer_n])
            if nn.use_biases[layer_n]
                nn.δl_δb[layer_n] -= δl_δb
            end

            if layer_n != 1
                δl_δa = transpose(nn.weights[layer_n]) * δl_δb
            end
        end
    end

    for layer_n in 1:length(nn.sizes)
        nn.weights[layer_n] += nn.learn_rates[layer_n] * nn.δl_δw[layer_n]
        fill!(nn.δl_δw[layer_n], 0.0)

        if nn.use_biases[layer_n]
            nn.biases[layer_n] += nn.learn_rates[layer_n] * nn.δl_δb[layer_n]
            fill!(nn.δl_δb[layer_n], 0.0)
        end
    end
end
