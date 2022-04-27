
# GUI not implemented yet
const display = terminal

# ["mnist", "balanced", "digits", "bymerge", "byclass"]
# 'letters' is broken
dataset_name = "mnist"

# must add to 100 (percent)
splits = [80, 20]

seed!(1)

epochs = [
    Epoch(
        batch_size = 10,
        shuffle = true)
    for i in 1:100]

model = Neural_Network(
    squared_error,
    784,

    # not implemented yet
    Float64,

    # [xavier, he]
    [xavier, xavier],

    # [z_score, demean, identity]
    [z_score, z_score],

    # [sigmoid, tanh, identity]
    # relu does not work yet
    # identity is untested
    [sigmoid, sigmoid],

    [0.01, 0.01],

    [28, length(mapping(dataset_name))],

    [true, true]
)
