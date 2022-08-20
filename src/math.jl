
using Distributions: Normal

# General

# exported by Base
# function identity(x)
#     return x
# end

function derivative(::typeof(identity))
    return function (x)
        return one(x)
    end
end

function mean(x)
    return sum(x) / length(x)
end

using Statistics: std, var, cov

# TODO: 'sum' is not type stable
# function variance(x, corrected = true) where T
#     return sum((x .- mean(x)) .^ 2) / (length(x) - corrected)
# end

# function standard_deviation(x, corrected = true)
#     return variance(x, corrected) ^ 0.5
# end

# Activation And Derivative

function sigmoid(x)
    one_x = one(x)
    return one_x / (one_x + exp(-x))
end

function derivative(f::typeof(sigmoid))
    return function (x)
        f_x = f(x)
        return f_x * (one(x) - f_x)
    end
end

function relu(x)
    return max(zero(x), x)
end

function derivative(::typeof(relu))
    return function (x)
        zero_x = zero(x)
        return x > zero_x ? one(x) : zero_x
    end
end

# exported by Base
# function tanh(x)
#     eˣ = exp.(x)
#     e⁻ˣ = inv.(eˣ)
#     return (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
# end

function derivative(f::typeof(tanh))
    return function (x)
        return one(x) - f(x) ^ 2
    end
end

function softmax(x)
    eˣ = exp(x - maximum(x, dims = 1))
    return eˣ / sum(eˣ, dims = 1)
end

function derivative(f::typeof(softmax))
    return function (xᵢ)
        # eˣᵢ = softmax(xᵢ)
        # return
    end
end

# Cost And Derivative

function squared_error(y, ŷ)
    return (y .- ŷ) .^ 2
end

function derivative(::typeof(squared_error))
    return function (y, ŷ)
        return 2 .* (ŷ .- y)
    end
end

# Normalization

function z_score(x)
    # return (x .- mean(x)) ./ standard_deviation(x)
    return (x .- mean(x)) ./ std(x)
end

function demean(x)
    return x .- mean(x)
end

# Initialization

function uniform()

end

function xavier(input_size)
    return Normal(0, input_size ^ -0.5)
end

function he(input_size)
    return 2 ^ 0.5 * xavier(input_size)
end

# Regularization

function weight_decay(x, λ)
    return λ * x
end

function l1(x, λ)
    return λ * abs(x)
end

function derivative(::typeof(l1))
    return function (x, λ)
        return λ * sign(x)
    end
end

function l2(x, λ)
    return λ / 2 * x ^ 2
end

function derivative(::typeof(l2))
    return weight_decay
end

# Inferential

function correlation_coefficient(x, y, corrected = true)
    return cov(x, y, corrected = corrected) / (std(x, corrected = corrected) * std(y, corrected = corrected))
end
