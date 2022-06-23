
function (layer::Dense)(x, activate)
    a = layer.w * x
    if !isnothing(layer.b)
        a .+= layer.b
    end

    map!(activate, a, a) # |> norm_func

    return a
end

function (model::NeuralNetwork)(x, layers_params)
    for (layer, layer_params) in zip(model.layers, layers_params)
        x = layer(x, layer_params.activate)
    end

    return x
end