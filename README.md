
## About

This project contains machine learning algorithms (currently only a neural network, but with more to come) written from scratch. It is intended for learning purposes; I do not expect it to be a viable package.

## Why Julia?

Julia is my favorite programming language for many reasons, but I think that its appeal is best stated by the founders [here](https://julialang.org/blog/2012/02/why-we-created-julia/).


## Instructions

Run an example script directly or create your own script using this project as a package. Using an 'EMNIST' dataset for the first time will download the datasets into the parent directory, which take several minutes. If you run into problems, stop the program, delete the 'emnist' folder, and start again.

Example scripts:
- neural_network.jl
    - Feed-forward multilayer perceptron
    - Run with command line arguments
        - ["concise", "verbose", "pretrained"]
- regression.jl
    - Simple linear regression

Note: Julia is JAOT (just ahead of time) compiled, meaning that 1) the initial compilation of packages will take a few seconds and 2) each method call with new argument types will take time to compile specialized code for those types. After this occurs, Julia will run quite swiftly.


## Configurable Features

- Seed
    - [random, deterministic]
- Data
    - Datasets
        - EMNIST
            - ["mnist", "balanced", "digits", "bymerge", "byclass", "letters"]
                - Note: "letters" is broken
    - Preprocessing
        - [z_score, demean, identity]
        - Splitting by percentage
- Epoch Parameters
    - Data shuffle boolean
    - Number of epochs
    - Loss function
        - [squared_error]
    - Batch size
        - stochastic -> 1
        - mini-batch -> 1 < x < n
        - batch -> n
    - Batch normalization
        - [z_score, demean, identity]
    - Layer Parameters
        - Learning rates
        - Activation functions
            - [sigmoid, tanh, relu, identity]
        - Regularization
            - Note: not implemented correctly
            - [weight_decay, l1, l2]
                - Note: 'weight_decay' is equivalent to 'l2' when not using adaptive gradients
                    - https://arxiv.org/pdf/1711.05101v3.pdf
- Model
    - Saving and loading
    - Feed-forward multilayer perceptron
        - Optimization
            - [backpropagation]
        - Weight initialization functions
            - [xavier, he]
                - Note: 'he' is untested
        - Sizes
        - Use biases boolean
    - Simple regression
        - Correlation coefficient
        - Optimization
            [analytic]

## Planned Features

- Graphical user interface
    - Use plotting library
    - Display inputs
    - User created inputs
- Models
    - Neural networks
        - [GenerativeAdversarialNetwork, ConvolutionalNeuralNetwork]
    - Regression
        - Multiple
    - Decision tree
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
- Visualizations
- Optimizers
    - ADAM
- Regulizers
    - Early stopping
    - Dropout?
- Parallel processing/threading
- Automatic differentiation?
- Documentation
- Pretty printing
- Error and value checking
- Hyperparameter optimization


## Known Bugs

- EMNIST 'letters' dataset labels are incorrect
- Regularization not implemented correctly


## To Do

- More
    - Functional style
    - Generic
- Optimize
    - Type stability
    - Time & space complexity


## References

3Blue1Brown. (2017, November 3). Backpropagation calculus | Chapter 4, Deep learning [Video]. Youtube. https://www.youtube.com/watch?v=tIeHLnjs5U8

Bezanson, J., Edelman, A., Karpinski, S., & Shah, V. B. (2017). Julia: A fresh approach to numerical computing. SIAM Review, 59(1), 65–98. https://doi.org/10.1137/141000671

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
