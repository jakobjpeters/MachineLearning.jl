
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
        println("    Split ", i, "\tAccuracy: ", round(accuracy, digits = 4), "\t\tLoss: ", round(loss, digits = 8))
    end
    println()
end

function gui(dataset, epochs, model)
    error("GUI not implemented yet.")
end
