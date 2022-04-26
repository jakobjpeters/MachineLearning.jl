
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

model = FFN(
    squared_error,
    784,

    # not implemented yet
    Float64,
    LayerHyperparameters((
        # [xavier, he]
        weight_init_funcs = [xavier, xavier],

        learn_rates = [0.01, 0.01],

        # [z_score, demean, identity]
        norm_funcs = [z_score, identity],

        # [sigmoid, tanh]
        # relu does not work yet
        activ_funcs = [sigmoid, sigmoid],

        use_biases = [true, true],

        sizes = [28, length(mapping(dataset_name))]
    ))
)
