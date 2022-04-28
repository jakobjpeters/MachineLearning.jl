
Current Features
    Random seed
    Data
        Dataset loading
            EMNIST - https://www.nist.gov/itl/products-and-services/emnist-dataset
                mnist, balanced, digits, bymerge, byclass
        Preprocessing
        Splitting into train, validate, holdout, etc sets
    Epochs
        Batch size
            stochastic = 1
            mini-batch = 1 < x < length(inputs)
            batch = length(inputs)
        Data shuffle
    Model
        Cost function
            squared_error
    Layers
        Learning rates
        Weight initialization functions
            xavier
            he (untested)
        Activation functions
            sigmoid
            tanh
            relu
            identity (untested)
        Use biases
        Sizes

Planned Features:
    Graphical user interface
        Use plotting library
        Display inputs
        User created inputs
    Machine learning models
        Generative adversarial network
        Regression
    Datasets
    Functions
        Cost functions
        Activation functions
        Weight initialization functions
        Prediction
            Max value
            Max n values
            Cutoff values
                ROC analysis?
    Floating point precision
    Visualizations
    Optimizers
    Regulizers
    Save & load models
    Parallel processing/threading
    Automatic differentiation?
    Documentation
    Pretty printing
    Batch normalization
    Error and value checking

Bugs:
    EMNIST 'letters' dataset labels are incorrect
    'print_data' doesn't work

Other TODOs:
    More
        Functional style
        Generic
    Optimize
        Type stability
        Runtime & space complexity
            Vectorize feed-forward and backpropagation
        types.jl
            Better parametric typing
            Some lines are too long
        emnist.jl
            My first Julia code -> needs overhaul
            Eliminate redundant data
                Delete zip and gzip or delete zip and load gzip dataset on the fly
