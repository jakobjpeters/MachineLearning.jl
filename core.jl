
function predict!(nn, input)
    nn.preallocs.as[begin] = input

    for layer_n in 1:length(nn.sizes)
        nn.preallocs.as[layer_n] = nn.norm_funcs[layer_n](nn.preallocs.as[layer_n])

        nn.preallocs.zs[layer_n] = nn.params.ws[layer_n] * nn.preallocs.as[layer_n]
        if nn.use_biases[layer_n]
            nn.preallocs.zs[layer_n] += nn.params.bs[layer_n]

        nn.preallocs.as[layer_n + 1] = nn.activ_funcs[layer_n](nn.preallocs.zs[layer_n])
        end
    end
end

function assess!(nn, inputs, labels)
    correct = 0
    error = 0.0

    for (input, label) in zip(inputs, labels)
        predict!(nn, input)

        if issetequal(findall(x -> x > 0.5, nn.preallocs.as[end]), label)
            correct += 1
        end

        error += mean(nn.cost_func(nn.preallocs.as[end], label))
    end

    return correct / length(labels), error / length(labels)
end

function backpropagate!(nn, inputs, labels)
    for (input, label) in zip(inputs, labels)
        predict!(nn, input)

        δl_δa = deriv(nn.cost_func, nn.preallocs.as[end], label)

        for layer_n in reverse(1:length(nn.sizes))
            δl_δb = δl_δa .* deriv(nn.activ_funcs[layer_n], nn.preallocs.zs[layer_n])

            nn.preallocs.δl_δw[layer_n] -= δl_δb * transpose(nn.preallocs.as[layer_n])
            if nn.use_biases[layer_n]
                nn.preallocs.δl_δb[layer_n] -= δl_δb
            end

            if layer_n != 1
                δl_δa = transpose(nn.params.ws[layer_n]) * δl_δb
            end
        end
    end

    for layer_n in 1:length(nn.sizes)
        nn.params.ws[layer_n] += nn.learn_rates[layer_n] * nn.preallocs.δl_δw[layer_n]
        fill!(nn.preallocs.δl_δw[layer_n], 0.0)

        if nn.use_biases[layer_n]
            nn.params.bs[layer_n] += nn.learn_rates[layer_n] * nn.preallocs.δl_δb[layer_n]
            fill!(nn.preallocs.δl_δb[layer_n], 0.0)
        end
    end
end
