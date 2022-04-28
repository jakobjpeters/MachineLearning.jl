
# Math Functions

function _error(prediction, label)
    error = deepcopy(prediction)

    for i in label
        error[i] -= 1
    end

    return error
end

function identity(x)
    return deepcopy(x)
end

function deriv(f::typeof(identity), x)
    return ones(size(x))
end

# Activation Functions And Derivatives

function sigmoid(xs)
    return 1 ./ (1 .+ exp.(-xs))
end

function deriv(f::typeof(sigmoid), xs)
    fxs = f(xs)
    return fxs .* (1 .- fxs)
end

function relu(xs)
    return max.(0, xs)
end

function deriv(::typeof(relu), xs)
    return map(x -> x > 0 ? 1 : 0, xs)
end

function tanh(xs)
    exs = exp.(xs)
    e_xs = inv.(exs)
    return (exs .- e_xs) ./ (exs .+ e_xs)
end

function deriv(f::typeof(tanh), xs)
    return 1 .- f(xs) .^ 2
end

function softmax(xs)
    # esp_xs = exp.(xs)
    # return esp_xs ./ sum(esp_xs)
end

function deriv(::typeof(softmax), xs)

end

# Error Functions And Derivatives

function squared_error(prediction, label)
    return _error(prediction, label) .^ 2
end

function deriv(f::typeof(squared_error), prediction, label)
    return 2 * _error(prediction, label)
end

# Normalization functions

function z_score(xs)
    return (xs .- mean(xs)) / std(xs)
end

function demean(xs)
    return xs .- mean(xs)
end

# Initialization Functions

function zero()

end

function uniform()

end

function xavier(input_size)
    return Normal(0, input_size ^ -0.5)
end

function he(input_size)
    return 2 ^ 0.5 * xavier(input_size)
end
