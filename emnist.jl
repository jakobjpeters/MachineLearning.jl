
# TODO: clean up
# TODO: improve memory usage for datasets

const datasets = ["mnist", "balanced", "digits", "letters", "bymerge", "byclass"]
const dir = pwd() * "/emnist/"

function decompress(in_name, out_name)
    f = GZip.open(in_name)
    write(out_name, read(f))
    close(f)

    return nothing
end

function read_uint8(f_name, offset)
    f = open(f_name)
    data::Array{UInt8, 1} = deleteat!(read(f), 1:offset)
    close(f)

    return convert.(Int64, data)
end

function read_string(f_name)
    f = open(f_name)
    data::Array{String, 1} = readlines(f)
    close(f)
    return data
end

function read_images(f_name, offset)
    data = read_uint8(f_name, offset)
    images = reshape(data, (28 ^ 2, :))
    return [images[:, i] for i in 1:size(images)[end]]
end

function read_labels(f_name, offset)
    return [[label + 1] for label in read_uint8(f_name, offset)]
end

function mapping(dataset)
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

    return nothing
end

# broken
function fix_letters_labels(key)
    path = pwd() * "emnist/decompressed/letters_" * key * "_labels.bin"
    labels = read_uint8(path, 0)
    f = open(path, "w")
    write(f, labels .- 1)
    close(f)

    return nothing
end

function load_emnist(name)
    dir = pwd() * "/emnist/decompressed/"

    inputs = read_images(dir * name * "_train_images.bin", 16)
    append!(inputs, read_images(dir * name * "_test_images.bin", 16))
    labels = read_labels(dir * name * "_train_labels.bin", 8)
    append!(labels, read_labels(dir * name * "_test_labels.bin", 8))

    return Data(inputs, labels)
end

function init()

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

            # if dataset == "letters"
                # fix_letters_labels("test")
            # end

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

    return nothing
end

init()