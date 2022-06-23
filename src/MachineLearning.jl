    
module MachineLearning

using InteractiveUtils: @which, @code_warntype

include("math.jl")
include("types.jl")
include("utilities.jl")
include("interface.jl")
include("core.jl")

export
    # math.jl
    identity, derivative, mean, # general
    sigmoid, relu, tanh, # activation
    softmax, squared_error, # cost
    z_score, demean, # standardization
    xavier, he, # weight initialization
    weight_decay, l1, l2, # regularization

    # interface.jl
    terminal,

    # core.jl
    train!,
    assess

    # types.jl
    Assessment, Data # 
    LayerParameters, EpochParameters, Cache,
    NeuralNetwork, Dense, SimpleLinearRegression, # models

    # utilities.jl
    load_dataset

end # module
