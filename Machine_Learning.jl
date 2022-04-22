    
module Machine_Learning

# testing
using InteractiveUtils
using BenchmarkTools

# external
import GZip
import ZipFile
import Distributions
import Statistics
import StatsBase
import Random
import LinearAlgebra

# GUI


# internal
include("functions.jl")
include("emnist.jl")
include("types.jl")
include("core.jl")
include("interface.jl")
include("config.jl")
include("print.jl")

# fix
__init__()

Random.seed!(config.seed)
config.display(config.dataset, config.model_hparams, config.layer_hparams)

end
