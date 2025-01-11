
module MachineLearning

import Base: show
using Distributions: Normal
using Downloads: download
using GZip: gzopen
using .Iterators: drop
using LinearAlgebra: axpy!, BLAS.gemm!, muladd
using Luxor: Drawing, Point, box, finish, setcolor
using Random: shuffle!
using ZipFile: Reader

include("neural_networks.jl")
include("math.jl")
include("core.jl")
include("emnist.jl")

export
    EMNIST, NeuralNetwork, balanced, by_class, by_merge, digits, letters, mnist,
    load_emnist, squared_error, relu, sigmoid, test_emnist, train_classifier, train!, z_score

end # MachineLearning
