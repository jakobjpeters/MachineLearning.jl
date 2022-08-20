
using Flux
using MachineLearning

function main()
    dataset = load_dataset("mnist", z_score)
    datasets = split_dataset(dataset, [80, 20])

    epochs = 10

    opt = Descent(0.01)
    loss = Flux.mse

    model = Chain(
        Flux.Dense(784, 100, Flux.sigmoid),
        Flux.Dense(100, 10, Flux.sigmoid)
    )

    # precompile
    ps = Flux.params(model)
    gs = gradient(() -> loss(model(datasets[begin].x), datasets[begin].y), ps)
    Flux.Optimise.update!(opt, ps, gs)

    ps = Flux.params(model)

    @time for epoch in 1:epochs
        @time gs = gradient(() -> loss(model(datasets[begin].x), datasets[begin].y), ps)
        @time Flux.Optimise.update!(opt, ps, gs)
        println(epoch)
    end
end

main()