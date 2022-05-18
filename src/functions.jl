
# General

function identity(x)
    return x
end

function deriv(f::typeof(identity), x::T) where T
    return ones(T, size(x))
end

function mean(xᵢ)
    return sum(xᵢ) / length(xᵢ)
end

# Activation And Derivative

function sigmoid(xᵢ)
    return 1 ./ (1 .+ exp.(-xᵢ))
end

function deriv(f::typeof(sigmoid), xᵢ)
    f_xᵢ = f(xᵢ)
    return f_xᵢ .* (1 .- f_xᵢ)
end

function relu(xᵢ)
    return max.(0, xᵢ)
end

function deriv(::typeof(relu), xᵢ)
    return map(x -> x > 0 ? 1 : 0, xᵢ)
end

function tanh(xᵢ)
    eˣᵢ = exp.(xᵢ)
    e⁻ˣᵢ = inv.(eˣᵢ)
    return (eˣᵢ .- e⁻ˣᵢ) ./ (eˣᵢ .+ e⁻ˣᵢ)
end

function deriv(f::typeof(tanh), xᵢ)
    return 1 .- f(xᵢ) .^ 2
end

function softmax(xᵢ)
    # esp_xᵢ = exp.(xᵢ)
    # return esp_xᵢ ./ sum(esp_xᵢ)
end

function deriv(::typeof(softmax), xᵢ)

end

# Cost And Derivative

function squared_error(Y, Ŷ)
    return (Y .- Ŷ) .^ 2
end

function deriv(f::typeof(squared_error), Y, Ŷ)
    return 2 * (Y .- Ŷ)
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
