
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

const dataset = load_dataset(
    dataset_name,

    # preprocessing function
    z_score,

    # dataset split percentages
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

    # cost function
    squared_error,

    # input size
    # cannot be changed
    # TODO: make automatic
    784,

    # precision
    # not implemented yet
    Float64,

    # weight initialization functions
    # [xavier, he]
    [xavier, xavier],

    # normalization functions
    # [z_score, demean, identity]
    [z_score, z_score],

    # activation functions
    # [sigmoid, tanh, identity]
    # identity is untested
    # relu does not work yet
    [sigmoid, sigmoid],

    # learn rates
    [0.01, 0.01],

    # layer sizes
    # last layer cannot be changed
    [28, length(mapping(dataset_name))],

    # use biases
    [true, true]
)
