
Current Features:
    Parametric configuration and hyperparameters
        EMNIST datasets - https://www.nist.gov/itl/products-and-services/emnist-dataset
            mnist, balanced, digits, bymerge, byclass
        Random seed choice
        Model
            Epochs
            Shuffle inputs
            Batch size
            Cost functions
                squared_error
        Layers
            Learning rates
            Weight initialization functions
                xavier
                he
            Normalization functions
                z_score
                demean
            Activation functions
                sigmoid
                tanh
            Use biases
            Sizes

Planned Features:
    Graphical user interface
        Display inputs
        User created inputs
    More
        Machine learning models
            Generative adversarial network
            Regression
        Datasets
        Cost functions
        Activation functions
        Weight initialization functions
    Parameterized floating point precision
    Parameterized prediction function
        Max value
        Max n values
        Cutoff values
            ROC analysis?
    Visualizations
    Optimizers
    Regulizers
    Save & load models
    Parallel processing
    Automatic differentiation?
    Documentation
    Pretty printing
    Batch normalization
    Error and value checking

Bugs:
    EMNIST letters dataset labels are incorrect
    RELU doesn't work

Other TODOs:
    More
        Functional style
        Generic
    Optimize
        Type stability
        Runtime & space complexity
        core.jl
            Functions currently pass around and access fields of massive struct
                Hard to read
                Potentially need better design
        types.jl
            Better parametric typing
            Some lines are too long
        emnist.jl
            My first Julia code -> needs overhaul
            Eliminate redundant data
                Delete zip and gzip or delete zip and load gzip dataset on the fly
