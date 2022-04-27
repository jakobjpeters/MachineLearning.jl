
const datasets = ["mnist", "balanced", "digits", "letters", "bymerge", "byclass"]
const dir = pwd() * "/emnist/"

function decompress(in_name::String, out_name::String)
    f = GZip.open(in_name)
    write(out_name, read(f))
    close(f)
end

function read_uint8(f_name::String, offset::Integer)
    f = open(f_name)
    data::Array{UInt8, 1} = deleteat!(read(f), 1:offset)
    close(f)

    return convert.(Int64, data)
end

function read_string(f_name::String)
    f = open(f_name)
    data::Array{String, 1} = readlines(f)
    close(f)
    return data
end

function read_images(f_name::String, offset::Integer)
    data = read_uint8(f_name, offset)
    # change to only use reshape? mapslices()?
    images = reshape(data, (28 ^ 2, :))
    return [images[:, i] for i in 1:size(images)[end]]
end

function read_labels(f_name::String, offset::Integer, output_size) # output size?
    return [[label + 1] for label in read_uint8(f_name, offset)]
end

function mapping(dataset::String)
    mapping = read_string(dir * "decompressed/" * dataset * "_mapping.txt")
    map::Dict{UInt8, Char} = Dict()

    for line in mapping
        map[parse(UInt8, split(line, " ")[1]) + 1] = Char(parse(UInt8, split(line, " ")[2]))
    end
    return map
end

function fix_letters_map()
    path = pwd() * "emnist/decompressed/" * "letters_mapping.txt"
    mapping = read_string(path)

    map::Array{Tuple{Int64, UInt8}, 1} = []
    # why is key UInt8 and val Int?
    for line in mapping
        push!(map, (parse(UInt8, split(line, " ")[1]) - 1, parse(UInt8, split(line, " ")[2])))
    end

    f = open(path, "w")
    for key_val in map
        write(f, string(key_val[1]) * " " * string(key_val[2]) * "\n")
    end
    close(f)
end

# broken
function fix_letters_labels(key)
    path = pwd() * "emnist/decompressed/letters_" * key * "_labels.bin"
    labels = read_uint8(path, 0)
    f = open(path, "w")
    write(f, labels .- 1)
    close(f)
end

# depreciate
function print_images(images, labels, map, ids::Array{T, 1}) where T <: Integer
    for id in ids
        print_image(images[id], map[labels[id]])
    end
end

function print_images(images, labels)
    for (image, label) in zip(images, labels)
        print_image(image, label)
    end
end

# depreciate
function print_image(image::Array{T, 1}, label::Char = ' ') where T <: Real
    # display(image)
    image = reshape(image, 28, 28)

    for row in 1:28
        for pixel in image[row, :]
            if pixel == minimum(image)
                print("_")
            elseif pixel > mean(image)
                print("O")
            else
                print("o")
            end
        end
        println()
    end
    println(label, "\n")
end

function load_emnist(name)
    dir = pwd() * "/emnist/decompressed/"

    inputs = read_images(dir * name * "_train_images.bin", 16)
    append!(inputs, read_images(dir * name * "_test_images.bin", 16))
    labels = read_labels(dir * name * "_train_labels.bin", 8, 10)
    append!(labels, read_labels(dir * name * "_test_labels.bin", 8, 10))

    return Data(inputs, labels)
end

function init_emnist()

    if !isdir("emnist/decompressed")
        mkpath("emnist/decompressed")
    end
    if !isdir("emnist/gzip")
        mkpath("emnist/gzip")
    end

    cd("emnist")
    if !isfile("emnist.zip")
        download("http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip", "emnist.zip")
        
        zip = ZipFile.Reader("emnist.zip")
        for f in zip.files
            touch(f.name)
            out = open(f.name, "w")
            write(out, read(f, String))
            close(out)
        end
        close(zip)
    end
    cd("..")

    for dataset in datasets
        if !isfile(dir * "decompressed/" * dataset * "_train_images.bin")
            decompress(dir * "gzip/emnist-" * dataset * "-train-images-idx3-ubyte.gz", dir * "decompressed/" * dataset * "_train_images.bin")
            # println(dataset * " train images decompressed")
        end

        if !isfile(dir * "decompressed" * dataset * "_test_images.bin")
            decompress(dir * "gzip/emnist-" * dataset * "-test-images-idx3-ubyte.gz", dir * "decompressed/" * dataset * "_test_images.bin")
            # println(dataset * " test images decompressed\n")
        end

        if !isfile(dir * "decompressed/" * dataset * "_train_labels.bin")
            decompress(dir * "gzip/emnist-" * dataset * "-train-labels-idx1-ubyte.gz", dir * "decompressed/" * dataset * "_train_labels.bin")

            if dataset == "letters"
                # fix_letters_labels("train")
            end

            # println(dataset * " train_labels decompressed\n")
        end

        if !isfile(dir * "decompressed/" * dataset * "_test_labels.bin")
            decompress(dir * "gzip/emnist-" * dataset * "-test-labels-idx1-ubyte.gz", dir * "decompressed/" * dataset * "_test_labels.bin")

            if dataset == "letters"
                # fix_letters_labels("test")
            end

            # println(dataset * " test_labels decompressed\n")
        end
        
        if !isfile(dir * "decompressed/" * dataset * "_mapping.txt")
            cp(dir * "gzip/emnist-" * dataset * "-mapping.txt", dir * "decompressed/" * dataset * "_mapping.txt")

            # if dataset == "letters"
            #     fix_letters_map()
            # end

            # println(dataset * " mapping decompressed\n")
        end
    end
end
