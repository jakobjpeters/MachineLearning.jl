
using Printf: @printf

# TODO: improve
# prints model assessment for each data split
function terminal(assessments)
    println("\nEpoch: ", length(assessments) - 1)

    header = "\tSplit\t\tCost"
    costs = assessments[end].costs
    accuracies = nothing

    if in(:accuracies, keys(assessments[end]))
        header *= "\t\tAccuracy"
        accuracies = assessments[end].accuracies
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
