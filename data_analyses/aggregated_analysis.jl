# julia --threads=8 --project="." "data_analyses/aggregated_analysis.jl"
using MultilevelGGMSampler
import DelimitedFiles, StatsBase, JLD2, CodecZlib, LinearAlgebra
import ProgressMeter

#region function
read_file(file) = DelimitedFiles.readdlm(file, '\t', Float64)

function read_rawdata(files)

    # rawdata = read_file(first(files))
    # prog = ProgressMeter.Progress(length(files) - 1, "aggregating raw data")
    # for i in 2:length(files)
    #     thisdata = read_file(files[i])
    #     rawdata = hcat(rawdata, thisdata)
    #     ProgressMeter.next!(prog)
    # end
    rawdata = ProgressMeter.@showprogress "aggregating raw data" map(read_file, files)
    return rawdata

end

function process_data_aggregated(rawdata)
    data_aggregated = reduce(hcat, rawdata)
    p, n = size(data_aggregated)
    k = 1
    prepare_data(data_aggregated ./ fourthroot(k))
end
function process_data_aggregated2_helper(data_one_person)
    scattermatrix = LinearAlgebra.Symmetric(StatsBase.scattermat(data_one_person, dims = 2))
    if LinearAlgebra.isposdef(scattermatrix)
        return Array(inv(scattermatrix))
    else
        return Array(LinearAlgebra.pinv(scattermatrix))
    end
end
function process_data_aggregated2(rawdata)
    # want: inverse of precision matrix
    # have: inverse of individual scatter matrices
    k = length(rawdata)
    n = size(first(rawdata), 2)
    aggregated_scattermat = inv(sum(process_data_aggregated2_helper(rawdata[i]) for i in eachindex(rawdata)))# .* n)
    prepare_data(aggregated_scattermat .* k^2, n * k)
end
#endregion

function main()
    root_dir_files = "/home/don/hdd/surfdrive/Shared/GIN/"

    @assert isdir(root_dir_files)
    # root_dir_files = "/run/media/don/HDD/surfdrive/Shared/GIN/"
    all_files = readdir(root_dir_files, join = true)

    k    = length(all_files)
    p    = 116#nothing

    test_run = false
    if !test_run
        n_iter   = 20_000 # 50_000
        n_warmup = 5_000
    else
        n_iter   = 15
        n_warmup = 5
    end

    postfix = test_run ? "test" : "run"
    results_dir = "/home/don/hdd/surfdrive/Postdoc/ABC/simulations/data_analyses/fixed_model8"
    # results_dir = "/home/don/hdd/surfdrive/Postdoc/ABC/simulations/data_analyses/results_aggregated_new_cholesky_k$(k)_test=$(test_run)"
    !isdir(results_dir) && mkdir(results_dir)
    @assert isdir(results_dir)

    @info "Starting analysis with" k, n_iter, n_warmup

    filename0 = joinpath(results_dir, "aggregated_data.jdl2")
    if !isfile(filename0)
        rawdata = read_rawdata(all_files)
        JLD2.jldsave(filename0, true; rawdata = rawdata)
    else
        rawdata = JLD2.jldopen(filename0)["rawdata"]::Vector{Matrix{Float64}}
    end
    pdata_1 = process_data_aggregated(rawdata)
    pdata_2 = process_data_aggregated2(rawdata)

    individual_structure = SpikeAndSlabStructure(threaded = false, method = MultilevelGGMSampler.CholeskySampling(), inv_method = MultilevelGGMSampler.CG_Inv())

    init_K = nothing
    init_G = nothing

    samples = MultilevelGGMSampler.sample_GGM(pdata_1,
        individual_structure,
        n_iter                      = n_iter,
        n_warmup                    = n_warmup,
        save_precmats               = false,
        save_graphs                 = true,
        init_K                      = init_K,
        init_G                      = init_G
    )

    filename1 = joinpath(results_dir, "aggregated_$(postfix)_1.jdl2")
    JLD2.jldsave(filename1, true; data = pdata_1, samples = samples)

    samples = MultilevelGGMSampler.sample_GGM(pdata_2,
        individual_structure,
        n_iter                      = n_iter,
        n_warmup                    = n_warmup,
        save_precmats               = false,
        save_graphs                 = true,
        init_K                      = init_K,
        init_G                      = init_G
    )

    filename2 = joinpath(results_dir, "aggregated_$(postfix)_2.jdl2")
    JLD2.jldsave(filename2, true; data = pdata_2, samples = samples)

    return nothing
end

main()