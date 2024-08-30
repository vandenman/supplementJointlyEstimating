using MultilevelGGMSampler
import DelimitedFiles, StatsBase, JLD2, CodecZlib, Dates
include("../utilities.jl")

#region functions
read_file(file) = DelimitedFiles.readdlm(file, '\t', Float64)

function read_and_prepare_data(files::AbstractVector, p = nothing)


    rawdata = read_file(first(files))
    n = size(rawdata, 2)
    p = size(rawdata, 1)
    k = length(files)

    sum_of_squares = Array{Float64, 3}(undef, p, p, k)
    sum_of_squares[:, :, 1] = StatsBase.scattermat(rawdata[1:p, :]; dims=2)
    for i in 2:length(files)
        rawdata = read_file(files[i])[1:p, :]
        if size(rawdata) != (p, n)
            # file 494 has one more observation than the others... why?
            # @show i, p, n
            # @assert size(rawdata) == (p, n)
        end
        sum_of_squares[:, :, i] = StatsBase.scattermat(rawdata; dims=2)
    end

    return (; n, p, k, sum_of_squares)
end
#endregion

function main(
        ;
        root_dir_files = "/home/don/hdd/surfdrive/Shared/GIN/",
        results_dir    = joinpath(pwd(), "data_analyses", "fits"),
        test_run::Bool = is_test_run()
    )

    postfix = test_run ? "test" : "run"
    filename = joinpath(results_dir, "$postfix.jld2")

    if isfile(filename)
        log_message("Exiting multilevel analysis because the \"$filename\" already exists. Rename or delete it if you want to run the analysis again.")
        return nothing
    end

    !isdir(results_dir) && mkdir(results_dir)
    @assert isdir(results_dir)
    @assert isdir(root_dir_files)
    all_files = readdir(root_dir_files, join = true)
    @assert !isempty(all_files)



    k        = length(all_files) # 724
    p        = nothing           # nothing = all

    if !test_run
        n_iter   = 20_000 # 50_000
        n_warmup = 15_000
    else
        n_iter   = 15
        n_warmup = 5
    end

    log_message("Preparing data files")

    selected_files = all_files[1:k]
    data = read_and_prepare_data(selected_files, p)
    (;n, p) = data

    log_message("Starting multilevel analysis with t = $n, p = $p, n = $k, n_iter = $n_iter, n_warmup = $n_warmup")

    individual_structure = SpikeAndSlabStructure(threaded = true, method = MultilevelGGMSampler.CholeskySampling(), inv_method = MultilevelGGMSampler.CG_Inv(), σ_spike = 0.05)

    samples = MultilevelGGMSampler.sample_MGGM(data,
        individual_structure,
        CurieWeissStructure();
        n_iter                      = n_iter,
        n_warmup                    = n_warmup,
        save_individual_precmats    = false,
        save_individual_graphs      = true,
        save_group_samples          = true
    )

    log_message("Saving results to $filename")

    JLD2.jldopen(filename, "w"; compress = true) do file

        for field in fieldnames(typeof(samples))
            group_name = string(field)
            value = getfield(samples, field)
            file[group_name] = value
        end
    end

    return nothing
end

main()
