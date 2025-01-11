
const zero, one, two = Float32.(0:2)

relu(x::Real) = max(zero, x)

sigmoid(x::Real) = one / (one + exp(-x))

squared_error(output, labels) = @. (labels - output) ^ two

derivative(::typeof(identity)) = _ -> one
derivative(::typeof(relu)) = x -> x > zero ? one : zero
derivative(f::typeof(sigmoid)) = function (x)
    f_x = f(x)
    f_x * (one - f_x)
end
derivative(::typeof(squared_error)) = (output, labels) -> two * (output .- labels)
derivative(f::typeof(tanh)) = x -> one - f(x) ^ two

function z_score(x)
    _length = Float32(length(x))
    demeaned = x .- Float32(sum(x)) / _length
    demeaned / √(Float32(sum(demeaned .^ two)) / (_length - one))
end
