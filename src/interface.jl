
using Printf: @printf

# TODO: improve
# prints model assessment for each data split
function terminal(assessment, epoch_n = 0)
    println("\nEpoch: ", epoch_n)

    header = "\tSplit\t\tCost"
    costs = assessment.costs
    accuracies = nothing

    if in(:accuracies, keys(assessment))
        header *= "\t\tAccuracy"
        accuracies = assessment.accuracies
    end
    println(header)

    for split_n in eachindex(costs)
        @printf("\t%s\t\t%.4f", split_n, costs[split_n])
        if !isnothing(accuracies)
            @printf("\t\t%.4f", accuracies[split_n])
        end
        println()
    end

    return nothing
end

function gui(config)
    return error("GUI not implemented yet.")
end

function gui(dataset, epoch, model)
    return error("GUI not implemented yet.")
end
