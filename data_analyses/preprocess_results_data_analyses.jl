using MultilevelGGMSampler
import StatsBase as SB, JLD2, CodecZlib, Printf, OnlineStats, LoopVectorization
using LinearAlgebra
import ProgressMeter, Distributions, Random#, DataFrames as DF, DelimitedFiles
include("../utilities.jl")

#region functions
function get_posterior_means(obj_file, means_file)

    if isfile(means_file)
        return JLD2.load(means_file)["means"]
    else

        object = JLD2.jldopen(obj_file) do file

            get_mean = x -> OnlineStats.mean(x.stats[1])

            stats     = file["stats"]
            samples_G = file["samples_G"]

            means_G = dropdims(SB.mean(samples_G, dims = 3), dims = 3)
            means_K = get_mean.(stats.K)
            means_μ = get_mean.(stats.μ)
            means_σ = get_mean( stats.σ)

            (; means_G, means_K, means_μ, means_σ)
        end

        JLD2.jldsave(means_file, true; means = object)
        return object

    end
end

function jdotavx(a, b)
    # based on https://juliasimd.github.io/LoopVectorization.jl/latest/examples/dot_product/
    s = zero(eltype(a))
    LoopVectorization.@turbo for i ∈ eachindex(a, b)
        s += a[i] * b[i]
    end
    s
end

#endregion


function main(
    ;
    results_dir    = joinpath(pwd(), "data_analyses", "fits"),
    test_run::Bool = is_test_run()
)

    test_run && !endswith(results_dir, "test") && (results_dir *= "_test")
    !isdir(results_dir) && error("Directory $results_dir does not exist. Run the multilevel analysis first.")

    obj_file = joinpath(results_dir, "multilevel.jld2")
    !isfile(obj_file) && error("File $obj_file does not exist. Run the multilevel analysis first.")

    # G_samples_file = joinpath(results_dir, "G_samples_tril.jld2")
    # if !isfile(G_samples_file)
    #     let
    #         file = JLD2.jldopen(obj_file)
    #         samps = file["samples"]
    #         indicator_samples_tril = samps.samples_G
    #         JLD2.jldsave(
    #             joinpath(results_dir, "G_samples_tril.jld2"), true;
    #             indicator_samples_tril = indicator_samples_tril
    #         )
    #     end
    # end

    # JLD2.jldopen(obj_file)

    means_file = joinpath(results_dir, "multilevel_means.jld2")
    means_obj  = get_posterior_means(obj_file, means_file)

    # sample posterior group samples
    no_group_samples = test_run ? 100 : 10_000
    rng = Random.default_rng()
    Random.seed!(rng, 1234)
    d = CurieWeissDistribution(means_obj.means_μ, means_obj.means_σ)
    s = Distributions.sampler(d)
    group_samples = Matrix{Int}(undef, length(d), no_group_samples)
    prog = ProgressMeter.Progress(size(group_samples, 2), showspeed = true);
    Threads.@threads for ik in axes(group_samples, 2)
        Distributions.rand!(rng, s, view(group_samples, :, ik))
        ProgressMeter.next!(prog)
    end

    rng = Random.default_rng()
    prior_σ_sampler = Distributions.sampler(Distributions.truncated(Distributions.Normal(), 0, Inf))
    prior_μ_sampler = Distributions.sampler(Distributions.Uniform(-1e100, 1e100)) # change this to new prior
    prior_μ_samples = similar(means_obj.means_μ)

    prior_group_samples = Matrix{Int}(undef, length(prior_μ_samples), no_group_samples) # not a BitArray because then threading is not possible
    prog = ProgressMeter.Progress(size(prior_group_samples, 2), showspeed = true)
    Threads.@threads for ik in axes(prior_group_samples, 2)
        prior_σ_sample = rand(rng, prior_σ_sampler)
        Distributions.rand!(rng, prior_μ_sampler, prior_μ_samples)

        d_prior = CurieWeissDistribution(prior_μ_samples, prior_σ_sample)
        Distributions.rand!(rng, d_prior, view(prior_group_samples, :, ik))
        ProgressMeter.next!(prog)
    end

    prior_edge_inclusion_probs = vec(SB.mean(prior_group_samples, dims = 2))

    edge_inclusion_probs       = vec(SB.mean(group_samples, dims = 2))
    thresholded_probs          = edge_inclusion_probs .> .5

    pairwise_on_counts         = zeros(Int, size(group_samples, 1), size(group_samples, 1))

    graph_edge_probs        = MultilevelGGMSampler.tril_vec_to_sym(edge_inclusion_probs, -1)
    graph_thresholded_probs = BitMatrix(MultilevelGGMSampler.tril_vec_to_sym(thresholded_probs, -1))

    group_samples_trans = permutedims(group_samples)
    prog = ProgressMeter.Progress(size(group_samples_trans, 2), showspeed = true)
    Threads.@threads for ip in axes(group_samples_trans, 2)
        for jp in axes(group_samples_trans, 2)
            pairwise_on_counts[ip, jp] = jdotavx(view(group_samples_trans, :, ip), view(group_samples_trans, :, jp))
        end
        ProgressMeter.next!(prog)
    end

    filename = joinpath(pwd(), results_dir, "multilevel_group_samples.jld2")
    log_message("Saving group-level information to $filename")

    JLD2.jldsave(filename, true;
        group_samples              = BitMatrix(group_samples),
        prior_group_samples        = BitMatrix(prior_group_samples),
        prior_edge_inclusion_probs = prior_edge_inclusion_probs,
        edge_inclusion_probs       = edge_inclusion_probs,
        thresholded_probs          = thresholded_probs,
        pairwise_on_counts         = pairwise_on_counts,
        graph_edge_probs           = graph_edge_probs,
        graph_thresholded_probs    = graph_thresholded_probs
    )

    # store individual graphs
    individual_graph_edge_probs        = means_obj.means_G
    individual_graph_thresholded_probs = individual_graph_edge_probs .> .5

    filename = joinpath(pwd(), results_dir, "multilevel_individual_samples.jld2")
    log_message("Saving group-level information to $filename")

    JLD2.jldsave(filename, true;
        individual_graph_edge_probs        = individual_graph_edge_probs,
        individual_graph_thresholded_probs = individual_graph_thresholded_probs
    )


end

main()