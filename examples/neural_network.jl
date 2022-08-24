
using MachineLearning
using Random: seed!

# testing
using InteractiveUtils: @which, @code_warntype

function concise()
    dataset = load_dataset("mnist", z_score)
    datasets = split_dataset(dataset, [80, 20])

    model = NeuralNetwork(
        squared_error,
        size(dataset.x, 1),
        [100, size(dataset.y, 1)],
        [sigmoid, tanh]
    )
    caches = init_caches(length(model.layers))
    layers_params = init_layers_params(length(model.layers), 0.01)

    regularizers = map(layer_params -> layer_params.regularizer, layers_params)
    terminal(assess(datasets, model, regularizers))

    @time for i in 1:10
        @time train!(model, datasets[begin], 10, layers_params, caches)
        @time terminal(assess(datasets, model, regularizers), i)
    end
end

function verbose()
    # remove for random seed
    seed!(1)

    # Dataset
    # EMNIST
    # ["mnist", "balanced", "digits", "bymerge", "byclass", "letters"]
    # "letters" is broken
    dataset_name = "mnist"

    # [z_score, demean, identity]
    preprocessor = z_score

    split_percentages = [80, 20]

    dataset = load_dataset(dataset_name, preprocessor)
    datasets = split_dataset(dataset, split_percentages)

    input_size = size(dataset.x, 1)
    output_size = size(dataset.y, 1)

    # Model
    # [xavier, he]
    # 'he' is untested
    w_inits = [xavier, xavier]

    layer_sizes = [100, output_size]

    # [true, false]
    use_biases = [true, false]

    # [tanh, sigmoid, identity]
    # 'identity' is untested
    activators = [sigmoid, tanh]

    # [squared_error]
    loss = squared_error

    # ["Neural_Network", "Linear"]
    model = NeuralNetwork(loss, input_size, layer_sizes, activators, w_inits, use_biases)


    # layer normalization
    # [z_score, demean, identity]
    # not currently "plugged in"
    # layer_normalizers = [identity, identity]

    # not implemented correctly
    regularizers = repeat([Regularizer(
        # [weight_decay, l1, l2]
        # default is "weight_decay"
        # untested
        weight_decay,

        # set to '0.0' for no regularization
        # untested
        0.0
    )], 2)

    learn_rates = [0.1, 0.01]

    layers_params = init_layers_params(length(model.layers), learn_rates, regularizers)#, layer_normalizers)


    # batch normalization
    # [z_score, demean, identity]
    batch_normalize = z_score

    n_epochs = 10
    batch_size = 10
    shuffle = true

    # pre-training
    # see 'interface.jl'
    terminal(assess(datasets, model, regularizers))

    caches = init_caches(length(model.layers))

    # main training loop
    @time for i in 1:n_epochs
        # see 'core.jl'
        @time train!(model, datasets[begin], batch_size, layers_params, caches, batch_normalize, shuffle)
        @time terminal(assess(datasets, model, regularizers), i)
    end
end

# enter a command-line argument from ["concise", "verbose"]
getfield(Main, Symbol(ARGS[begin]))()
