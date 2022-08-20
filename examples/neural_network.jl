
# internal
using MachineLearning

# external
using Random: seed!

# testing
using InteractiveUtils: @which, @code_warntype

function concise()
    dataset = load_dataset("mnist", z_score)
    datasets = split_dataset(dataset, [80, 20])

    model = NeuralNetwork(784, [100, 10])
    layers_params = init_layers_params(length(model.layers), 0.01, tanh)
    loss = squared_error

    # pre-training
    # see 'interface.jl'
    terminal(assess(datasets, model, loss, layers_params))

    caches = init_caches(length(model.layers))

    # main training loop
    @time for i in 1:10
        # see 'core.jl'
        @time train!(model, datasets[begin], 10, layers_params, loss, caches)
        @time terminal(assess(datasets, model, loss, layers_params), i)
    end
end

function verbose()
    # comment out for random seed
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

    # Model
    # [xavier, he]
    # 'he' is untested
    w_inits = [xavier, xavier]

    # TODO: make dynamic
    input_size = 784

    # the last layer size is determined by the number of classes
    # TODO: make last layer size dynamic
    layer_sizes = [100, 10]

    # [true, false]
    use_biases = [true, false]

    # ["Neural_Network", "Linear"]
    model = NeuralNetwork(input_size, layer_sizes, w_inits, use_biases)


    # Layers_Parameters
    # [tanh, sigmoid, identity]
    # 'identity' is untested
    activators = [sigmoid, tanh]

    # not implemented correctly
    # regularizers = repeat([Regularizer(
    #     # [weight_decay, l1, l2]
    #     # default is "weight_decay"
    #     # untested
    #     weight_decay,

    #     # set to '0.0' for no regularization
    #     # untested
    #     0.0
    # )], 2)

    learn_rates = [0.1, 0.01]

    # layer normalization
    # [z_score, demean, identity]
    # not currently "plugged in"
    # layer_normalizers = [identity, identity]

    layers_params = init_layers_params(length(model.layers), learn_rates, activators)#, layer_normalizers, regularizers)


    # Epoch
    # [squared_error]
    loss = squared_error

    # batch normalization
    # [z_score, demean, identity]
    batch_normalize = z_score

    n_epochs = 10
    batch_size = 10
    shuffle = true

    # pre-training
    # see 'interface.jl'
    terminal(assess(datasets, model, loss, layers_params))

    caches = init_caches(length(model.layers))

    # main training loop
    @time for i in 1:n_epochs
        # see 'core.jl'
        @time train!(model, datasets[begin], batch_size, layers_params, loss, caches, batch_normalize, shuffle)
        @time terminal(assess(datasets, model, loss, layers_params), i)
    end
end

# enter a command-line argument from ["concise", "verbose"]
getfield(Main, Symbol(ARGS[begin]))()
