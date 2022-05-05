
# About

This project contains machine learning algorithms (currently only a neural network, but with more to come) written from scratch. It is intended for learning purposes; I do not expect it to be a viable package. I intend to learn by developing more models, features, configurations, performance optimizations, documentation, etc.

# Why Julia?

Julia is my favorite programming language for many reasons, but I think that its appeal is best stated by the founders [here](https://julialang.org/blog/2012/02/why-we-created-julia/).


# Instructions

Edit 'config.jl' to your liking. When ready, run 'Machine_Learning.jl'. This will initially download and decompress 'EMNIST' datasets, which may take a few minutes. If you run into problems, delete the 'emnist' folder and start again.

Note: Julia is JAOT (just ahead of time) compiled, meaning that 1) the initial compilation of packages will take a few seconds and 2) each method call with new argument types will take time to compile specialized code for those types. In this project, compilation will be complete after the 1st epoch.


# Configurable Features

- Seed
    - Random
    - Set
- Data
    - Datasets
         - EMNIST
            - mnist
            - balanced
            - digits
            - bymerge
            - byclass
            - letters (broken)
    - Preprocessing
        - z_score
        - demean
        - identity
    - Splitting by percentage
- Epochs
    - Data shuffle boolean
    - Number of epochs
    - Cost function
        - squared_error
    - Batch size
        - stochastic = 1
        - mini-batch = 1 < x < length(inputs)
        - batch = length(inputs)
- Model
    - Feed-forward multilayer perceptron with backpropagation
        - Weight initialization functions
            - xavier
            - he (untested)
        - Sizes
        - Use biases boolean
        - Hyperparameters
            - Learning rates
            - Activation functions
                - sigmoid
                - tanh
                - relu
                - identity (untested)


# Planned Features

- Graphical user interface
    - Use plotting library
    - Display inputs
    - User created inputs
- Models
    - Generative adversarial network
    - Regression
    - Decision tree
    - Convolutional neural network
    - K-nearest neighbors
- Datasets
- Functions
    - Cost functions
    - Activation functions
    - Weight initialization functions
    - Prepropcessing
        - Principal component analysis
    - Prediction
        - Max value
        - Max n values
        - Cutoff values
            - ROC analysis?
- Floating point precision
- Visualizations
- Optimizers
- Regulizers
- Save & load models
- Parallel processing/threading
- Automatic differentiation?
- Documentation
- Pretty printing
- Batch normalization
- Error and value checking
- Cached intermediate arrays
- Easier to customize 'Epoch' and 'Hyperparameters'
- Hyperparameter optimization


# Bugs

- EMNIST 'letters' dataset labels are incorrect
- 'print_data' doesn't work


# To Do

- More
    - Functional style
    - Generic
- Optimize
    - Type stability
    - Time & space complexity
    - types.jl
        - Better parametric typing
        - Some lines are too long

# References

3Blue1Brown. (2017, November 3). Backpropagation calculus | Chapter 4, Deep learning [Video]. Youtube. https://www.youtube.com/watch?v=tIeHLnjs5U8

Bezanson, J., Edelman, A., Karpinski, S., & Shah, V. B. (2017). Julia: A fresh approach to numerical computing. SIAM Review, 59(1), 65–98. https://doi.org/10.1137/141000671

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
