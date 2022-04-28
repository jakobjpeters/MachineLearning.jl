
function terminal(config)
    println()
    for key in keys(config)
        println(key, ":")
        println(config[key])
        println()
    end
end

function terminal(dataset, epoch, model)
    println("\nEpoch: ", epoch)
    # mse not type stable
    for (i, data) in enumerate(dataset)
        accuracy, loss = assess!(model, data.inputs, data.labels)
        @printf("\tSplit: %s\tAccuracy: %.4f\tLoss: %.4f\n", i, accuracy, loss)
    end
    println()
end

function gui(config)
    return error("GUI not implemented yet.")
end

function gui(dataset, epoch, model)
    return error("GUI not implemented yet.")
end
