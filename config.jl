# TODO: make it so a single variable is used across the array
# 'letters' labels are fucked up

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
        weight_init_func = default_weight_init
    ),

    layer_hparams = (
        learn_rates = [0.01, 0.01, 0.01],
        norm_funcs = [z_score, identity, identity],
        activ_funcs = [sigmoid, sigmoid, sigmoid],
        use_biases = [true, true, true],
        sizes = [280, 28, 10]
    )
)