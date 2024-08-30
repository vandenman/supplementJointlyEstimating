using MultilevelGGMSampler
import JLD2, CodecZlib, LinearAlgebra
import ProgressMeter
include("../utilities.jl")


#region functions
function read_rawdata2(files)

    rawdata = ProgressMeter.@showprogress "aggregating raw data" map(read_file, files)
    return rawdata

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

function main(
    ;
    root_dir_files = "/home/don/hdd/surfdrive/Shared/GIN/",
    results_dir    = joinpath(pwd(), "data_analyses", "fits"),
    test_run::Bool = is_test_run()
)

    test_run && !endswith(results_dir, "test") && (results_dir *= "_test")
    !isdir(results_dir) && mkdir(results_dir)
    @assert isdir(results_dir)

    filename = joinpath(results_dir, "aggregated.jld2")

    if isfile(filename)
        log_message("Exiting aggregated analysis because the \"$filename\" already exists. Rename or delete it if you want to run the analysis again.")
        return nothing
    end

    @assert isdir(root_dir_files)
    all_files = readdir(root_dir_files, join = true)
    @assert !isempty(all_files)

    if test_run
        n        = 30
        n_iter   = 15
        n_warmup = 5
    else
        n    = length(all_files)
        n_iter   = 20_000
        n_warmup = 5_000
    end

    log_message("Starting aggregated analysis with n = $n, n_iter = $n_iter, n_warmup = $n_warmup")

    rawdata = read_rawdata2(all_files)
    pdata_2 = process_data_aggregated2(rawdata)

    individual_structure = SpikeAndSlabStructure(threaded = false, method = MultilevelGGMSampler.CholeskySampling(), inv_method = MultilevelGGMSampler.CG_Inv())

    samples = MultilevelGGMSampler.sample_GGM(pdata_2,
        individual_structure,
        n_iter                      = n_iter,
        n_warmup                    = n_warmup,
        save_precmats               = false,
        save_graphs                 = true
    )

    log_message("Saving results to $filename")

    JLD2.jldsave(filename, true; data = pdata_2, samples = samples)

    return nothing
end

main()