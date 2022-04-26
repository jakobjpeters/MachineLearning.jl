
const display = terminal
dataset_name = "mnist"
splits = [80, 20]
seed!(1)

epochs = [
    Epoch(
        batch_size = 10,
        shuffle = true)
    for i in 1:100]

model = FFN(
    ModelHyperparameters((
        cost_func = squared_error,
        input_size = 784,
        precision = Float64
    )),
    LayerHyperparameters((
        weight_init_funcs = [xavier, xavier],
        learn_rates = [0.01, 0.01],
        norm_funcs = [z_score, identity],
        activ_funcs = [sigmoid, sigmoid],
        use_biases = [true, true],
        sizes = [28, 10]
    ))
)
