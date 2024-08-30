using MultilevelGGMSampler
import JLD2, CodecZlib
include("../utilities.jl")

function main(
    ;
    root_dir_files = "/home/don/hdd/surfdrive/Shared/GIN/",
    results_dir    = joinpath(pwd(), "data_analyses", "fits"),
    test_run::Bool = is_test_run()
)

    test_run && !endswith(results_dir, "test") && (results_dir *= "_test")
    !isdir(results_dir) && mkdir(results_dir)
    @assert isdir(results_dir)

    filename = joinpath(results_dir, "individual.jld2")

    if isfile(filename)
        log_message("Exiting individual analysis because the \"$filename\" already exists. Rename or delete it if you want to run the analysis again.")
        return nothing
    end

    @assert isdir(root_dir_files)
    all_files = readdir(root_dir_files, join = true)
    @assert !isempty(all_files)

    max_n = 1

    if test_run
        n_iter   = 15
        n_warmup = 5
    else
        n_iter   = 20_000
        n_warmup = 5_000
    end

    !isdir(results_dir) && mkdir(results_dir)
    @assert isdir(results_dir)

    log_message("Starting individual analysis with n = $max_n, n_iter = $n_iter, n_warmup = $n_warmup")

    data = read_and_prepare_data(all_files[1:max_n])

    individual_structure = SpikeAndSlabStructure(threaded = false, method = MultilevelGGMSampler.CholeskySampling(), inv_method = MultilevelGGMSampler.CG_Inv(), σ_spike = 0.05)
    group_structure      = MultilevelGGMSampler.IndependentStructure()

    samples = MultilevelGGMSampler.sample_MGGM(data,
        individual_structure,
        group_structure,
        n_iter                      = n_iter,
        n_warmup                    = n_warmup,
        save_individual_precmats    = false,
        save_individual_graphs      = true
    )

    log_message("Saving results to $filename")

    JLD2.jldsave(filename, true; data = data, samples = samples)

    return nothing
end

main()