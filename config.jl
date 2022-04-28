
# random seed
seed!(1)

# display
# [terminal]
# GUI not implemented yet
const display = terminal

# EMNIST dataset name
# ["mnist", "balanced", "digits", "bymerge", "byclass"]
# 'letters' is broken
const dataset_name = "mnist"
const output_size = length(mapping(dataset_name))

const dataset = load_dataset(
    dataset_name,

    # preprocessing function
    z_score,

    # split percentages
    # must add to 100
    [80, 20]
)

const epochs = map(i -> Epoch(
    # batch size
    10,
    
    # shuffle data
    true
), 1:100)

model = Neural_Network(

    # model cost function
    squared_error,

    # model input size
    # cannot be changed
    # TODO: make automatic
    784,

    # model precision
    # not implemented yet
    Float64,

    # layer weight initialization functions
    # [xavier, he]
    # he is untested
    [xavier, xavier],

    # layer normalization functions
    # [z_score, demean, identity]
    # not useful with current data and architecture
    # not currently "plugged in"
    [identity, identity],

    # layer activation functions
    # [tanh, sigmoid, identity]
    # identity is untested
    # relu does not work yet
    [tanh, tanh],

    # layer learn rates
    [0.01, 0.01],

    # layer sizes
    # output_size cannot be changed
    [28, output_size],

    # use biases
    [true, true]
)
