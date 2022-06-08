
# General

function identity(x)
    return x
end

function derivative(::typeof(identity))
    return function (x::T) where T
        return one(T)
    end
end

function mean(xᵢ)
    return sum(xᵢ) / length(xᵢ)
end

# Activation And Derivative

function sigmoid(x::T) where T
    return one(T) / (one(T) + exp(-x))
end

function derivative(f::typeof(sigmoid))
    return function (x::T) where T
        f_x = f(x)
        return f_x * (one(T) - f_x)
    end
end

function relu(x::T) where T
    return max(zero(T), x)
end

function derivative(::typeof(relu))
    return function (x::T) where T
        return x > zero(T) ? one(T) : zero(T)
    end
end

function tanh(x)
    eˣ = exp(x)
    e⁻ˣ = inv(eˣ)
    return (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
end

function derivative(f::typeof(tanh))
    return function (x::T) where T
        return one(T) - f(x) ^ 2
    end
end

function softmax(x)
    # esp_x = exp(x)
    # return esp_x / sum(esp_x)
end

function derivative(::typeof(softmax))
    return function (x)

    end
end

# Cost And Derivative

function squared_error(Y, Ŷ)
    return (Y .- Ŷ) .^ 2
end

function derivative(::typeof(squared_error))
    return function (Y, Ŷ)
        return 2 * (Ŷ - Y)
    end
end

# Normalization

function z_score(xᵢ)
    m = mean(xᵢ)
    return (xᵢ .- m) / stdm(xᵢ, m)
end

function demean(xᵢ)
    return xᵢ .- mean(xᵢ)
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
