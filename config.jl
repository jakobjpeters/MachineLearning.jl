# TODO: make it so a single variable is used across the array
# 'letters' labels are wrong

# this file must be a valid nested named-tuple
config = (
    display = terminal,
    dataset = "mnist",
    seed = 1,

    model_hparams = (
        cost_func = squared_error,
        epochs = 100,
        input_size = 784,
        batch_size = 10,
        precision = Float64,
        shuffle = true,
    ),

    layer_hparams = (
        weight_init_funcs = [xavier, xavier],
        learn_rates = [0.01, 0.01],
        norm_funcs = [z_score, identity],
        activ_funcs = [sigmoid, sigmoid],
        use_biases = [true, true],
        sizes = [28, 10]
    )
)
