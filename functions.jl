
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

function deriv(f::typeof(relu), xs)
    return [x > 0 ? 1 : 0 for x in xs]
end

function tanh(xs)
    exs = exp.(xs)
    e_xs = exp.(-xs)
    return (exs .- e_xs) ./ (exs .+ e_xs)
end

function deriv(f::typeof(tanh), xs)
    return 1 .- f(xs).^2
end

function softmax(xs)
    exs = exp.(xs)
    return exs ./ sum(exs)
end

function deriv(f::typeof(softmax), xs)
    # TODO: fix
    return LinearAlgebra.Diagonal(xs) .- (xs * xs')
end

function identity(x)
    return x
end

function deriv(f::typeof(identity), x::T) where T<:AbstractArray
    return fill!(x, 1)
end

# Error Functions And Derivatives

# TODO: make it change var, not allocate
function squared_error(prediction, label)
    error = copy(prediction)

    for i in label
        error[i] -= 1
    end

    return error.^2
end

# TODO: make it change var, not allocate
function deriv(f::typeof(squared_error), prediction, label)
    error = copy(prediction)

    for i in label
        error[i] -= 1
    end

    return 2 * error
end

# Normalization functions

function z_score(xs)
    return (xs .- mean(xs)) / Statistics.std(xs)
end

function demean(xs)
    return xs .- mean(xs)
end

# Initialization functions
function zero()

end

function uniform()

end

function xavier(input_size)
    return Distributions.Normal(0, sqrt(1 / input_size))
end

function he(input_size)
    return Distributions.Normal(0, sqrt(2 / input_size))
end