
function (regularizer::Regularizer)(w)
    regularizer.regularize(w, regularizer.λ)
end

function linear(w, x, b = nothing)
    return isnothing(b) ? w * x : muladd.(w, x, b)
end

function linear(w, x, b::AbstractArray)
    return isnothing(b) ? w * x : muladd(w, x, b)
end

function (layer::Linear)(x)
    return linear(layer.w, x, layer.b)
end

function (layer::Dense)(x)
    a = linear(layer.w, x, layer.b)
    map!(layer.activate, a, a) # |> norm_func

    return a
end

function (model::NeuralNetwork)(x)
    for layer in model.layers
        x = layer(x)
    end

    return x
end
