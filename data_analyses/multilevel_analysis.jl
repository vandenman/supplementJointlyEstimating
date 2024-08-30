using MultilevelGGMSampler
import DelimitedFiles, StatsBase, JLD2, CodecZlib, Dates
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

    filename = joinpath(results_dir, "multilevel.jld2")

    if isfile(filename)
        log_message("Exiting multilevel analysis because the \"$filename\" already exists. Rename or delete it if you want to run the analysis again.")
        return nothing
    end

    @assert isdir(root_dir_files)
    all_files = readdir(root_dir_files, join = true)
    @assert !isempty(all_files)


    if test_run
        n        = 30
        n_iter   = 15
        n_warmup =  5
    else
        n        = length(all_files) # 724
        n_iter   = 20_000
        n_warmup = 15_000
    end

    log_message("Preparing data files")

    selected_files = all_files[1:n]
    data = read_and_prepare_data(selected_files)
    t = data.n
    p = data.p

    log_message("Starting multilevel analysis with t = $t, p = $p, n = $n, n_iter = $n_iter, n_warmup = $n_warmup")

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

    # save every results element as a separate field in the JLD2 file, so they can be loaded individually
    JLD2.jldopen(filename, "w"; compress = true) do file

        for field in fieldnames(typeof(samples))
            group_name = string(field)
            value = getfield(samples, field)
            file[group_name] = value
        end
        file["data"] = data
    end

    return nothing
end

main()
