
function linear(w, x, b = nothing)
    return isnothing(b) ? w * x : muladd(w, x, b)
end

function (layer::Linear)(x)
    return linear(layer.w, x, layer.b)
end

function (layer::Dense)(x, activate)
    a = linear(layer.w, x, layer.b)
    map!(activate, a, a) # |> norm_func

    return a
end

function (model::NeuralNetwork)(x, activators)
    for (layer, activate) in zip(model.layers, activators)
        x = layer(x, activate)
    end

    return x
end
