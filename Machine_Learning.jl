    
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
include("emnist.jl")
include("types.jl")
include("core.jl")
include("interface.jl")
include("config.jl")
include("print.jl")

# fix
init_emnist()

seed!(config.seed)
config.display(config.dataset, config.model_hparams, config.layer_hparams)

end
