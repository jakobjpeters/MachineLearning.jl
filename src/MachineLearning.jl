    
module Machine_Learning

# Testing
import InteractiveUtils: @which

# External

# Files
import GZip
import ZipFile
import TOML

# Math
import Distributions: Normal
import Statistics: stdm
import Random: shuffle!, seed!
import Printf: @printf
import LinearAlgebra: BLAS.gemm!, axpy!

# GUI


# Internal
include("functions.jl")
include("emnist.jl")
include("utilities.jl")
include("types.jl")
include("core.jl")
include("interface.jl")

function main()
    config, display, dataset, epoch_param, num_epochs, model, caches = load_config()

    display(config)

    # pre-trained
    assessments = [Assessment(assess!(dataset, model, epoch_param.cost_func, epoch_param.layer_param, caches))]
    display(assessments)

    # main training loop
    # see 'core.jl'
    @time train_model!(epoch_param, model, caches, dataset, assessments, display, num_epochs)

    return
end

main()

end
