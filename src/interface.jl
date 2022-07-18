
using Printf: @printf

# prints model assessment for each data split
function terminal(assessments::AbstractArray{Assessment})
    println("\nEpoch: ", length(assessments) - 1)
    for (epoch_n, assessment) in enumerate(zip(last(assessments).accuracies, last(assessments).costs))
        @printf("\tSplit: %s\tAccuracy: %.4f\tCost: %.4f\n", epoch_n, assessment[begin], assessment[end])
    end
    println()

    return
end

function gui(config)
    return error("GUI not implemented yet.")
end

function gui(dataset, epoch, model)
    return error("GUI not implemented yet.")
end
