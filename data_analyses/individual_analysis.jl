using MultilevelGGMSampler
import DelimitedFiles, StatsBase, JLD2, CodecZlib
include("../utilities.jl")

#region function
read_file(file) = DelimitedFiles.readdlm(file, '\t', Float64)

function read_and_prepare_data(files, p = nothing)

    # TODO: do not make so many copies...

    rawdata = read_file(first(files))
    n = size(rawdata, 2)
    # if isnothing(p)
    p = size(rawdata, 1)
    # end
    k = length(files)
    # sum_of_squares = Vector{Matrix{Float64}}(undef, k)
    # sum_of_squares[1] = StatsBase.scattermat(rawdata[1:p, :]; dims=2)
    # for i in 2:length(files)
    #     rawdata = read_file(files[i])[1:p, :]
    #     @assert size(rawdata) == (p, n)
    #     sum_of_squares[i] = StatsBase.scattermat(rawdata; dims=2)
    # end
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
    test_run::Bool = get_test_run()
)

    postfix = test_run ? "test" : "run"
    filename = joinpath(results_dir, "$postfix.jld2")

    if isfile(filename)
        log_message("Exiting multilevel analysis because the \"$filename\" already exists. Rename or delete it if you want to run the analysis again.")
        return nothing
    end

    @assert isdir(root_dir_files)
    all_files = readdir(root_dir_files, join = true)

    max_k = 1

    if !test_run
        n_iter   = 20_000
        n_warmup = 5_000
    else
        n_iter   = 15
        n_warmup = 5
    end

    !isdir(results_dir) && mkdir(results_dir)
    @assert isdir(results_dir)

    postfix = test_run ? "test" : "run"
    filename = joinpath(results_dir, "individual_analaysis_$postfix.jld2")
    if isfile(filename)
        log_message("Exiting multilevel analysis because the \"$filename\" already exists. Rename or delete it if you want to run the analysis again.")
        return nothing
    end


    @info "Starting analysis with" max_k, n_iter, n_warmup

    data = read_and_prepare_data(all_files[1:max_k], 116)

    individual_structure = SpikeAndSlabStructure(threaded = false, method = MultilevelGGMSampler.CholeskySampling(), inv_method = MultilevelGGMSampler.CG_Inv(), σ_spike = 0.05)
    group_structure      = MultilevelGGMSampler.IndependentStructure()

    samples = MultilevelGGMSampler.sample_GGM(data,
        individual_structure,
        group_structure,
        n_iter                      = n_iter,
        n_warmup                    = n_warmup,
        save_precmats               = false,
        save_graphs                 = true
    )

    filename = joinpath(results_dir, "$(postfix)_k=$k.jdl2")
    JLD2.jldsave(filename, true; data = data, samples = samples)

    return nothing
end

main()