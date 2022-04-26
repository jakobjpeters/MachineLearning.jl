    
module Machine_Learning

# Testing
using InteractiveUtils
using BenchmarkTools

# External

# Emnist
import GZip
import ZipFile

# Math
import Distributions: Normal
import Statistics: mean, std
import Random: shuffle!, seed!

# GUI


# Internal
include("functions.jl")
include("types.jl")
include("core.jl")
include("emnist.jl")
include("print.jl")
include("interface.jl")
include("config.jl")

init_emnist()

display(dataset_name, splits, model, epochs)

end
