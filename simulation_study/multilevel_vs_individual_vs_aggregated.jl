using MultilevelGGMSampler, StatsBase
import Distributions, Random, ProgressMeter
import LinearAlgebra
import JLD2, CodecZlib
import OrderedCollections
using DataFrames
using Chain: @chain
import AlgebraOfGraphics as AOG
import CairoMakie
import SwarmMakie
import Printf, Dates
include("../utilities.jl")


#region simulation functions

function simulate_data(rng, n, p, k, rawdata = true, μs_mean = 2.0, μs_data_percentage = 100, σ = 0.1)

    ne = MultilevelGGMSampler.p_to_ne(p)

    if isnan(μs_mean)# == :data_based
        path = joinpath(pwd(), "simulation_study", "data_results_to_simulate_with.jld2")
        obj = JLD2.jldopen(path)
        avg_density = mean(MultilevelGGMSampler.tril_to_vec(obj["group_graph_edge_probs"], -1))
        group_G_tril = rand(Distributions.Bernoulli(avg_density), ne)

        if iszero(μs_data_percentage)
            μs = zeros(ne)
        else

            μs_data_proportion = μs_data_percentage / 100

            μ_incl_mean, μ_incl_sd =  obj["μ_if_present"] .* μs_data_proportion
            μ_excl_mean, μ_excl_sd =  obj["μ_if_absent"]  .* μs_data_proportion

            d_incl = Distributions.truncated(Distributions.Normal(μ_incl_mean, μ_incl_sd), 0.0, Inf)
            d_excl = Distributions.truncated(Distributions.Normal(μ_excl_mean, μ_excl_sd), -Inf, 0.0)
            μs = [ifelse(isone(gₑ), rand(d_incl), rand(d_excl)) for gₑ in group_G_tril]
        end

    else
        @assert μs_mean isa Number
        group_G_tril = rand(Distributions.Bernoulli(.3), ne)
        μs = (2 .* group_G_tril .- 1) .* μs_mean .+ randn(ne)
    end

    groupstructure = CurieWeissStructure(; μ = μs, σ = σ)

    πGW = MultilevelGGMSampler.GWishart(p, 3.0)
    data, parameters = MultilevelGGMSampler.simulate_hierarchical_ggm(rng, n, p, k, πGW, groupstructure, rawdata)
    return data, parameters, groupstructure, group_G_tril

end

process_data_multilevel(data, n, p, k) = size(data) == (p, n, k) ? prepare_data(data) : prepare_data(data, n)

function process_data_individual(data, n, p, k)
    if size(data, 3) == k
        return [prepare_data(data[:, :, ik]) for ik in 1:k]
    else
        return [prepare_data(data[:, :, ik], n) for ik in 1:k]
    end
end

function process_data_aggregated(data, n, p, k)
    data_aggregated = reshape(data, (p, n * k))
    # since
    # reshape(data, (p, n * k)) ≈ reduce(hcat, eachslice(data, dims = 3))
    prepare_data(data_aggregated ./ fourthroot(k))
end
function process_data_aggregated2_helper(data_one_person)
    scattermatrix = LinearAlgebra.Symmetric(scattermat(data_one_person, dims = 2))
    if LinearAlgebra.isposdef(scattermatrix)
        return Array(inv(scattermatrix))
    else
        return Array(LinearAlgebra.pinv(scattermatrix))
    end
end
function process_data_aggregated2(data, n, p, k)
    # want: inverse of precision matrix
    # have: inverse of individual scatter matrices
    aggregated_scattermat = inv(sum(process_data_aggregated2_helper(data[:, :, i]) for i in 1:k))# .* n)
    prepare_data(aggregated_scattermat .* k^2, n * k)
end

# function process_data_aggregated2(data, n, p, k)
#     # want: inverse of precision matrix
#     # have: inverse of individual scatter matrices
#     aggregated_scattermat = inv(mean(process_data_aggregated2_helper(data[:, :, i]) for i in 1:k))# .* n)
#     prepare_data(aggregated_scattermat .* k, n * k)
# end

function compute_confusion_matrix(observed, predicted, threshold = .5)
    @assert length(observed) == length(predicted)
    confusion_matrix = zeros(Int, 2, 2)
    for i in eachindex(observed, predicted)
        confusion_matrix[observed[i] + 1, (predicted[i] < threshold ? 1 : 2)] += 1
    end
    return confusion_matrix
end

function get_filename(runs_dir, n, p, k, method, μs_mean, r)
    return joinpath(runs_dir, "n=$(n)_p=$(p)_k=$(k)_method=$(string(method))_mu_mean=$(μs_mean)_r=$(r).jld2")
end
#endregion

#region plotfunctions

function link_axes!(fig_axes)
    for j in 2:length(fig_axes)
        linkaxes!(fig_axes[1], fig_axes[j])
    end
end

function plot_sample_stats(run_pdata, parameters, ik = 1,
        fig_sample_stats = Figure(),
        add_xlab = true, add_ylab = true, link = true)

    true_K_tril_participant_ik = MultilevelGGMSampler.tril_to_vec(@view parameters.K[:, :, ik])

    sample_stats_K_tril = (
        multilevel  = MultilevelGGMSampler.tril_to_vec(inv(         run_pdata.multilevel.sum_of_squares[:, :, ik]        ./ run_pdata.multilevel.n)),
        individual  = MultilevelGGMSampler.tril_to_vec(inv(dropdims(run_pdata.individual[ik].sum_of_squares,  dims = 3) ./ run_pdata.individual[ik].n)),
        # aggregated  = MultilevelGGMSampler.tril_to_vec(inv(dropdims(run_pdata.aggregated.sum_of_squares,      dims = 3) ./ run_pdata.aggregated.n)),
        aggregated2 = MultilevelGGMSampler.tril_to_vec(inv(dropdims(run_pdata.aggregated2.sum_of_squares,     dims = 3) ./ run_pdata.aggregated2.n))
    )

    # fig_sample_stats = Figure()
    fig_axes = []
    for (i, (key, result)) in enumerate(pairs(sample_stats_K_tril))
        ax = Axis(fig_sample_stats[1, i], title = Printf.@sprintf("%s: %.3f", string(key), cor(true_K_tril_participant_ik, result)))
        retrieval_plot!(ax, true_K_tril_participant_ik, result)
        push!(fig_axes, ax)
    end
    link && link_axes!(fig_axes)
    add_ylab && AOG.Label(fig_sample_stats[:, 0], "Sample precision", rotation = pi / 2, tellheight = false)
    add_xlab && AOG.Label(fig_sample_stats[2, 1:length(sample_stats_K_tril)], "True precision", tellwidth = false)

    return fig_sample_stats

end

function get_K_tril_participant_ik(run_results, ik = 1)
    return (
        multilevel  =     map(x->mean(x.stats[1]), run_results.multilevel.stats.K)[:, ik],
        individual  = vec(map(x->mean(x.stats[1]), run_results.individual[ik].stats.K)),
        # aggregated  = vec(map(x->mean(x.stats[1]), run_results.aggregated.stats.K)),
        aggregated2 = vec(map(x->mean(x.stats[1]), run_results.aggregated2.stats.K))
    )
end

function plot_k_tril(run_results, parameters, ik::Integer = 1)

    true_K_tril_participant_ik = MultilevelGGMSampler.tril_to_vec(@view parameters.K[:, :, ik])
    est_K_tril_participant_ik = get_K_tril_participant_ik(run_results, ik)

    fig_axes = []
    fig_retrieval_K_participant_ik = Figure()
    for (i, (key, result)) in enumerate(pairs(est_K_tril_participant_ik))
        ax = Axis(fig_retrieval_K_participant_ik[1, i], title = Printf.@sprintf("%s: %.3f", string(key), cor(true_K_tril_participant_ik, result)))
        retrieval_plot!(ax, true_K_tril_participant_ik, result)
        push!(fig_axes, ax)
    end
    link_axes!(fig_axes)
    AOG.Label(fig_retrieval_K_participant_ik[:, 0], "Posterior means", rotation = pi / 2, tellheight = false)
    AOG.Label(fig_retrieval_K_participant_ik[2, 1:length(est_K_tril_participant_ik)], "True precision", tellwidth = false)

    return fig_retrieval_K_participant_ik

end


function plot_avg_k_tril(run_results, parameters, fig_retrieval_K = Figure(),
    add_ylab = true, add_xlab = true)

    k = size(parameters.K, 3)
    avg_true_K_tril = MultilevelGGMSampler.tril_to_vec(dropdims(mean(parameters.K, dims = 3), dims = 3))
    avg_K_tril = (
        multilevel  = vec(   mean(map(x->mean(x.stats[1]), run_results.multilevel.stats.K), dims = 2)),
        individual  =    mean(vec(map(x->mean(x.stats[1]), run_results.individual[ik].stats.K)) for ik in 1:k),
        aggregated  =         vec(map(x->mean(x.stats[1]), run_results.aggregated.stats.K)),
        aggregated2 =         vec(map(x->mean(x.stats[1]), run_results.aggregated2.stats.K))
    )

    # fig_retrieval_K = Figure()
    fig_axes = []
    for (i, (key, result)) in enumerate(pairs(avg_K_tril))
        ax = Axis(fig_retrieval_K[1, i], title = Printf.@sprintf("%s\nρ = %.3f", string(key), cor(avg_true_K_tril, result)))
        retrieval_plot!(ax, avg_true_K_tril, result)
        push!(fig_axes, ax)
    end
    link_axes!(fig_axes)
    add_ylab && AOG.Label(fig_retrieval_K[:, 0], "Posterior means", rotation = pi / 2, tellheight = false)
    add_xlab && AOG.Label(fig_retrieval_K[2, 1:length(avg_K_tril)], "True precision", tellwidth = false)

    return fig_retrieval_K

end

function get_samples_G(run_results)
    samples_G_multilevel  = run_results.multilevel.samples_G
    # samples_G_aggregated  = run_results.aggregated.samples_G
    samples_G_aggregated2 = run_results.aggregated2.samples_G
    samples_G_individual  = similar(samples_G_multilevel)

    k = size(run_results.multilevel.samples_G, 2)
    for ik in 1:k
        samples_G_individual[:, ik, :] .= run_results.individual[ik].samples_G[:, 1, :]
    end

    samples_G = (
        multilevel  = samples_G_multilevel,
        individual  = samples_G_individual,
        # aggregated  = samples_G_aggregated,
        aggregated2 = samples_G_aggregated2
    )

    means_G = map(samples_G) do x
        dropdims(SB.mean(x, dims = 3), dims = 3)
    end

    return samples_G, means_G
end


function plot_incl_probs_vs_K_hat(run_results, parameters, ik = 1, use_true_K_hat = false,
    fig_incl_probs_vs_K_hat = Figure(size = (1000, 500)),
    add_ylab = true, add_xlab = true, add_legend = true)

    _, means_G = get_samples_G(run_results)

    true_g_vec_participant_ik = MultilevelGGMSampler.tril_to_vec(parameters.G[:, :, ik], -1)
    est_prob_1_participant_ik = map(means_G) do means
        if isone(size(means, 2))
            return vec(means)
        else
            return means[:, ik]
        end
    end

    est_K_tril_participant_ik = get_K_tril_participant_ik(run_results, ik)

    cols = reshape(Colors.distinguishable_colors(7)[[2, 3, 4, 6]], 2, 2)

    fig_axes = []
    # fig_incl_probs_vs_K_hat = Figure(size = (1000, 500))
    # (i, (key, prob_1, k_tril)) = collect(enumerate(zip(keys(est_prob_1_participant_ik),
                                                        #   est_prob_1_participant_ik,
                                                        #   est_K_tril_participant_ik)))[3]
    for (i, (key, prob_1, k_tril)) in enumerate(zip(keys(est_prob_1_participant_ik),
                                                          est_prob_1_participant_ik,
                                                          est_K_tril_participant_ik))


        ax = Axis(fig_incl_probs_vs_K_hat[1, i], title = Printf.@sprintf("%s", key))

        cols_scatter = map(zip(true_g_vec_participant_ik, prob_1 .> .5)) do (i1, i2)
            cols[i1 + 1, i2 + 1]
        end

        # this line is rather ugly
        k_tril2 = if use_true_K_hat
            MultilevelGGMSampler.tril_to_vec(parameters.K[:, :, ik], -1)
        else
            MultilevelGGMSampler.tril_to_vec(MultilevelGGMSampler.tril_vec_to_sym(k_tril), -1)
        end
        scatter!(ax, k_tril2, prob_1, color = cols_scatter)
        push!(fig_axes, ax)
    end

    link_axes!(fig_axes)

    add_ylab && AOG.Label(fig_incl_probs_vs_K_hat[:, 0], "Posterior inclusion probability", rotation = pi / 2, tellheight = false)
    add_xlab && AOG.Label(fig_incl_probs_vs_K_hat[2, 1:length(est_K_tril_participant_ik)], "Posterior means", tellwidth = false)

    if add_legend
        legend_elems = [MarkerElement(color = col, marker = :circle, markersize = 15) for col in vec(cols)]
        Legend(fig_incl_probs_vs_K_hat[1, 5], legend_elems, ["true negative", "false negative", "false positive", "true positive"])
    end

    return fig_incl_probs_vs_K_hat

end

function precision_to_partial(x, p)

    @assert size(x, 1) == p * (p + 1) ÷ 2

    result = similar(x, p * (p + 1) ÷ 2, size(x, 2))

    diagonal_indices = Vector{Int}(undef, p)#diagind(p, p)
    diagonal_indices[1] = 1
    for i in 2:length(diagonal_indices)
        diagonal_indices[i] = diagonal_indices[i - 1] + p + 2 - i
    end
    # @assert x[diagonal_indices, 2] ≈ diag(obs_precs[:, :, 2])
    for ik in axes(x, 2)

        raw_c_idx = 1

        for ip in 1:p
            diag_i = x[diagonal_indices[ip], ik]
            for jp in ip:p
                diag_j = x[diagonal_indices[jp], ik]
                raw_ij = x[raw_c_idx, ik]

                # @show raw_c_idx, offset, ip, jp, diagonal_indices[ip], diagonal_indices[jp]
                if raw_c_idx == diagonal_indices[ip] || raw_c_idx == diagonal_indices[jp]
                    result[raw_c_idx, ik] = 1.0
                else
                    result[raw_c_idx, ik] = -raw_ij / sqrt(diag_i * diag_j)
                end
                @assert !isnan(result[raw_c_idx, ik])
                raw_c_idx += 1
            end
        end
        @assert findall(isone, view(result, :, ik)) == diagonal_indices
    end
    non_diagonal_indices = setdiff(axes(result, 1), diagonal_indices)
    result_no_ones = result[non_diagonal_indices, :]
    result, result_no_ones
end

function plot_incl_probs_vs_parcor_hat(run_results, parameters, ik = 1, use_true_K_hat = false,
    fig_incl_probs_vs_K_hat = Figure(size = (1000, 500)),
    add_ylab = true, add_xlab = true, add_legend = true)

    _, means_G = get_samples_G(run_results)

    true_g_vec_participant_ik = MultilevelGGMSampler.tril_to_vec(parameters.G[:, :, ik], -1)
    est_prob_1_participant_ik = map(means_G) do means
        if isone(size(means, 2))
            return vec(means)
        else
            return means[:, ik]
        end
    end

    # NOTE: this is not entirely correct... it should happen on the level of the partial
    # correlation directly!
    est_K_tril_participant_ik = get_K_tril_participant_ik(run_results, ik)

    p = size(parameters.G, 1)
    est_parcor_tril_participant_ik = map(est_K_tril_participant_ik) do est_K_tril
        vec(precision_to_partial(est_K_tril, p)[2])
    end

    cols = reshape(Colors.distinguishable_colors(7)[[2, 3, 4, 6]], 2, 2)

    fig_axes = []
    for (i, (key, prob_1, parcor_tril)) in enumerate(zip(keys(est_prob_1_participant_ik),
                                                          est_prob_1_participant_ik,
                                                          est_parcor_tril_participant_ik))


        ax = Axis(fig_incl_probs_vs_K_hat[1, i], title = Printf.@sprintf("%s", key))#,
                #   limits = ((-1, 1), (0, 1)))

        cols_scatter = map(zip(true_g_vec_participant_ik, prob_1 .> .5)) do (i1, i2)
            cols[i1 + 1, i2 + 1]
        end

        # this line is rather ugly
        parcor_tril2 = if use_true_K_hat
            precision_to_partial(MultilevelGGMSampler.tril_to_vec(parameters.K[:, :, ik], -1), p)[2]
        else
            parcor_tril
        end
        scatter!(ax, parcor_tril2, prob_1, color = cols_scatter)
        push!(fig_axes, ax)
    end

    link_axes!(fig_axes)

    add_ylab && AOG.Label(fig_incl_probs_vs_K_hat[:, 0], "Posterior inclusion probability", rotation = pi / 2, tellheight = false)
    add_xlab && AOG.Label(fig_incl_probs_vs_K_hat[2, 1:length(est_K_tril_participant_ik)], "Posterior means", tellwidth = false)

    if add_legend
        legend_elems = [MarkerElement(color = col, marker = :circle, markersize = 15) for col in vec(cols)]
        Legend(fig_incl_probs_vs_K_hat[1, 5], legend_elems, ["true negative", "false negative", "false positive", "true positive"])
    end

    return fig_incl_probs_vs_K_hat

end


function plot_individual_aucs(run_results, parameters, fig_individuallevel = Figure(),
    add_title = true, add_legend = true)

    true_G_vec = vec(mapslices(x->tril_to_vec(x, -1), parameters.G, dims = 1:2))

    # TODO: this should be done once, and cached!
    _, means_G = get_samples_G(run_results)

    k = size(parameters.G, 3)

    auc_res = map(means_G) do mean_g
        # if the length mathches, we have predictions for each individual
        # otherwise we simply repeat the predictions of the aggregated model for each individual
        x = length(mean_g) == length(true_G_vec) ? vec(mean_g) : repeat(vec(mean_g), k)
        compute_roc_auc(true_G_vec, x)
    end


    ax = Axis(fig_individuallevel[1, 1], xlabel = "False positive rate", ylabel = "True positive rate")
    ablines!(ax, 0, 1; color = :grey)
    lines_obj = [lines!(ax, auc.fpr, auc.tpr) for auc in auc_res]

    if add_legend
        legend_txt = [Printf.@sprintf("%s AUC: %.3f", string(method), auc[3]) for (method, auc) in pairs(auc_res)]
        axislegend(ax, lines_obj, legend_txt, position = :rb, framevisible = false, backgroundcolor = :transparent)
    end
    add_title && AOG.Label(fig_individuallevel[0, :], "ROC curves predicting individual-level graph structures", tellwidth = false)
    return fig_individuallevel

end

function plot_group_aucs(run_results, parameters, group_G_tril, add_avg_multilevel = false,
    fig_grouplevel = Figure(), add_title = true)

    post_mean_μ = vec(mean(run_results.multilevel.groupSamples.μ, dims = 2))
    post_mean_σ = mean(run_results.multilevel.groupSamples.σ)
    posterior_CurieWeiss = CurieWeissDistribution(post_mean_μ, post_mean_σ)

    _, means_G = get_samples_G(run_results)

    preds_G_group = OrderedCollections.OrderedDict()
    # model implied group-level inclusion probabilities
    preds_G_group[:multilevel]  = exp.(MultilevelGGMSampler.compute_log_marginal_probs_approx(posterior_CurieWeiss))
    # average all individual inclusion probabilities
    if add_avg_multilevel
        preds_G_group[:multilevel2] = vec(mean(means_G.multilevel, dims = 2))
    end
    # average all individual inclusion probabilities (2 lines)
    preds_G_group[:individual]  = vec(mean(means_G.individual, dims = 2))
    # aggregation happened at the raw data
    # preds_G_group[:aggregated]  = vec(means_G.aggregated)
    preds_G_group[:aggregated2] = vec(means_G.aggregated2)

    colors = if add_avg_multilevel
        Makie.wong_colors()[[1 ; 5 ; 2:4]]
    else
        Makie.wong_colors()
    end

    auc_res_group = OrderedCollections.OrderedDict(
        method => compute_roc_auc(group_G_tril, mean_g)
        for (method, mean_g) in preds_G_group
    )

    # compute inlcusion probabilities implied by the true model
    true_CurieWeiss = CurieWeissDistribution(parameters.μ, parameters.σ)
    true_probs      = exp.(MultilevelGGMSampler.compute_log_marginal_probs_approx(true_CurieWeiss))
    max_roc = compute_roc_auc(group_G_tril, true_probs).auc # best you can do here!

    # different metric for evaluating, correlation between true and estimated inclusion probs
    cors_true_probs_vs_ests = map(values(preds_G_group)) do preds
        cor(true_probs, preds)
    end

    # fig_grouplevel = Figure()
    ax = Axis(fig_grouplevel[1, 1], xlabel = "False positive rate", ylabel = "True positive rate")
    ablines!(ax, 0, 1; color = :grey)
    lines_obj = [
        lines!(ax, auc.fpr, auc.tpr, color = colors[i])
        for (i, auc) in enumerate(values(auc_res_group))
    ]
    auc = auc_res_group[:aggregated]

    lines(
        [0.0; repeat(auc[1][1:end - 1], inner = 2); auc[1][end]],
        [0.0; repeat(auc[2][2:end], inner = 2);     auc[2][end]]
    )

    legend_txt = [Printf.@sprintf("%s AUC: %.3f", string(method), auc[3]) for (method, auc) in auc_res_group]
    axislegend(ax, lines_obj, legend_txt, position = :rb, framevisible = false, backgroundcolor = :transparent)
    add_title && AOG.Label(fig_grouplevel[0, :], "ROC curves predicting group-level graph structure", tellwidth = false)

    return fig_grouplevel, cors_true_probs_vs_ests, max_roc

end

function compute_confusion_matrices(run_results, parameters)

    true_G_vec = vec(mapslices(x->tril_to_vec(x, -1), parameters.G, dims = 1:2))
    _, means_G = get_samples_G(run_results)
    run_confusion_tables = map(means_G) do mean_G
        x = length(mean_G) == length(true_G_vec) ? vec(mean_G) : repeat(vec(mean_G), k)
        confusion_matrix = compute_confusion_matrix(true_G_vec, x)
        truecorrect = (confusion_matrix[1] + confusion_matrix[4]) / sum(confusion_matrix)
        confusion_matrix, truecorrect
    end
    return run_confusion_tables
end


#endregion

function run_simulation(results_dir, runs_dir, overwrite::Bool = false, test_run::Bool = true)

    test_run_prefix = test_run ? "test_" : ""
    results_file = joinpath(results_dir, "$(test_run_prefix)aggregated_results.jld2")
    if isfile(results_file) && !overwrite

        log_message("Loading results from disk")
        return JLD2.jldopen(results_file)["res"]

    end

    if test_run

        log_message("Running test run simulation")

        ps = (20, )
        ns = (50, )
        ks = (50, )
        μs = (0, 50, 100, ) # percentage of data-based μs to use
        reps = 1
        n_iter   = 200
        n_warmup = 100

    else

        log_message("Running full simulation")

        ps = (40, )
        ns = (50, 100, 500, 1000)
        ks = (50, 100, 300, 500)
        μs = (0, 10, 30, 50, 100, ) # percentage of data-based μs to use
        reps = 5
        n_iter   = 5_000
        n_warmup = 3_000

    end

    methods = (:multilevel, :individual, :aggregated2)

    ss_multilevel  = SpikeAndSlabStructure(; threaded = true,  method = MultilevelGGMSampler.CholeskySampling(), inv_method = MultilevelGGMSampler.CG_Inv())#, σ_spike = σ_spike)
    ss_individual  = SpikeAndSlabStructure(; threaded = true,  method = MultilevelGGMSampler.CholeskySampling(), inv_method = MultilevelGGMSampler.CG_Inv())#, σ_spike = σ_spike)
    ss_aggregated  = SpikeAndSlabStructure(; threaded = false, method = MultilevelGGMSampler.CholeskySampling(), inv_method = MultilevelGGMSampler.CG_Inv())#, σ_spike = σ_spike)

    rng = Random.default_rng()

    sim_opts0 = Iterators.product(ns, ps, ks, μs,          1:reps)
    sim_opts  = Iterators.product(ns, ps, ks, μs, methods, 1:reps)
    nsim = length(sim_opts)
    global_progress = true#false
    prog = ProgressMeter.Progress(nsim, "Overall progress", enabled = global_progress)

    results_df = DataFrame(
        n      = Vector{Int}(undef, nsim),
        p      = Vector{Int}(undef, nsim),
        k      = Vector{Int}(undef, nsim),
        μ      = Vector{Float64}(undef, nsim),
        r      = Vector{Int}(undef, nsim),
        method = Vector{Symbol}(undef, nsim),

        true_G_vec          = Vector{BitVector}(undef, nsim),
        group_G_tril        = Vector{BitVector}(undef, nsim),
        est_G_vec           = Vector{Matrix{Float64}}(undef, nsim),
        est_G_tril_group    = Vector{Vector{Float64}}(undef, nsim),
        means_K             = Vector{Matrix{Float64}}(undef, nsim),
        post_mean_μ         = Vector{Union{Nothing, Vector{Float64}}}(undef, nsim),
        post_mean_σ         = Vector{Union{Nothing, Float64}}(undef, nsim)
    )
    fill!(results_df.post_mean_μ, nothing)
    fill!(results_df.post_mean_σ, nothing)

    # this takes a bit long...
    filename_datasets = joinpath(runs_dir, "datasets.jdl2")
    if isfile(filename_datasets)
        log_message("Loading simulated datasets from disk")
        datasets = JLD2.load(filename_datasets)["datasets"]
        if length(sim_opts0) != length(datasets) # TODO: this check is suboptimal, should just loop over sim_opts0
            log_message("Simulating additional datasets")
            ProgressMeter.@showprogress for (n, p, k, μ, r) in sim_opts0
                key = (n, p, k, μ, r)
                if !haskey(datasets, key)
                    seed = Base.hash(key)
                    Random.seed!(seed)
                    datasets[key] = simulate_data(rng, n, p, k, true, NaN, μ)
                end
            end
            log_message("Saving datasets to disk")
            JLD2.jldsave(filename_datasets, true; datasets = datasets)
        end
    else

        log_message("Simulating datasets")
        temp = typeof(simulate_data(rng, ns[1], ps[1], ks[1], true, NaN, μs[1]))
        datasets = Dict{typeof(first(sim_opts0)), temp}()
        ProgressMeter.@showprogress for (n, p, k, μ, r) in sim_opts0
            key = (n, p, k, μ, r)
            seed = Base.hash(key)
            Random.seed!(seed)
            datasets[key] = simulate_data(rng, n, p, k, true, NaN, μ)
        end
        log_message("Saving datasets to disk")
        JLD2.jldsave(filename_datasets, true; datasets = datasets)
    end

    prog = ProgressMeter.Progress(nsim, "Overall progress", enabled = global_progress)
    log_message("Starting analyses")
    for (it, (n, p, k, μ, method, r)) in enumerate(sim_opts)


        filename = get_filename(runs_dir, n, p, k, method, μ, r)
        if isfile(filename)
            row = JLD2.load(filename)["res"]
            results_df[it, :] = row
            ProgressMeter.next!(prog)
            continue
        end

        results_df.n[it]      = n
        results_df.p[it]      = p
        results_df.k[it]      = k
        results_df.method[it] = method
        results_df.μ[it]      = μ
        results_df.r[it]      = r

        data, params, groupstructure, group_G_tril = datasets[(n, p, k, μ, r)]

        # results_df.parameters[it]   = params
        results_df.true_G_vec[it]   = BitVector(vec(mapslices(x->tril_to_vec(x, -1), params.G, dims = 1:2)))
        results_df.group_G_tril[it] = group_G_tril

        if method == :multilevel
            pdata = process_data_multilevel(data, n, p, k)
            res = sample_MGGM(pdata, ss_multilevel, groupstructure; rng = rng, n_iter = n_iter, n_warmup = n_warmup, save_individual_precmats = false, verbose = !global_progress);
            samples_G = res.samples_G
            means_K   = map(x->mean(x.stats[1]), res.stats.K)

            # results_df.pdata[it]        = pdata
            # results_df.samples_G[it]    = samples_G
            results_df.est_G_vec[it]    = dropdims(mean(samples_G, dims = 3), dims = 3)
            results_df.means_K[it]      = means_K
            results_df.post_mean_μ[it]  = vec(mean(res.groupSamples.μ, dims = 2))
            results_df.post_mean_σ[it]  = mean(res.groupSamples.σ)

            posterior_CurieWeiss = CurieWeissDistribution(results_df.post_mean_μ[it], results_df.post_mean_σ[it])
            results_df.est_G_tril_group[it] = exp.(MultilevelGGMSampler.compute_log_marginal_probs_approx(posterior_CurieWeiss))

        elseif method == :individual

            pdata = process_data_multilevel(data, n, p, k)
            res = sample_MGGM(pdata, ss_individual, MultilevelGGMSampler.IndependentStructure(); rng = rng, n_iter = n_iter, n_warmup = n_warmup, save_individual_precmats = false, verbose = !global_progress);
            samples_G = res.samples_G
            means_K   = map(x->mean(x.stats[1]), res.stats.K)

            # results_df.pdata[it]        = pdata
            # results_df.samples_G[it]    = samples_G
            results_df.est_G_vec[it]    = dropdims(mean(samples_G, dims = 3), dims = 3)
            results_df.means_K[it]      = means_K

            # ne = MultilevelGGMSampler.p_to_ne(p)
            results_df.est_G_tril_group[it] = vec(mean(results_df.est_G_vec[it], dims = 2))


        elseif method == :aggregated || method == :aggregated2
            pdata = if method == :aggregated
                process_data_aggregated(data, n, p, k)
            elseif method == :aggregated2
                process_data_aggregated2(data, n, p, k)
            end
            res = sample_GGM(pdata, ss_aggregated; rng = rng, n_iter = n_iter, n_warmup = n_warmup, save_precmats = false, verbose = !global_progress);
            samples_G = res.samples_G
            means_K   = map(x->mean(x.stats[1]), res.stats.K)

            # results_df.pdata[it]        = pdata

            g_means = vec(mean(samples_G, dims = 3))
            results_df.est_G_vec[it]    = reshape(repeat(g_means, k), length(g_means), k)
            results_df.means_K[it]      = reshape(means_K, p * (p + 1) ÷ 2, 1)
            results_df.est_G_tril_group[it] = results_df.est_G_vec[it][:, 1]

        end

        row = results_df[it, :]
        JLD2.jldsave(filename, true; res = row)

        ProgressMeter.next!(prog)

    end


    log_message("Saving results to $results_file")
    JLD2.jldsave(results_file, true; res = results_df)

    return results_df

end

function save_figures(simulation_results_df, figures_dir)

    thresholds = range(0.0, stop = 1.0, length = 1001)
    simulation_results_df.auc_individual = ProgressMeter.@showprogress map(eachrow(simulation_results_df)) do row
        isnothing(row.est_G_vec) && return nothing
        compute_roc_auc(row.true_G_vec, vec(row.est_G_vec), thresholds)
    end

    simulation_results_df.auc_grouplevel = map(eachrow(simulation_results_df)) do row
        compute_roc_auc(row.group_G_tril, row.est_G_tril_group, thresholds)
    end

    simulation_results_df.auc_indiv = map(x->x.auc, simulation_results_df.auc_individual)
    simulation_results_df.auc_group = map(x->x.auc, simulation_results_df.auc_grouplevel)


    # plot n against auc individual. rows = k, cols = μ
    method_names_rn = AOG.renamer([:multilevel => "Multilevel", :individual => "Individual", :aggregated2 => "Aggregated"])
    # n_names_rn = AOG.renamer(["100" => "N = 100", "300" => "N = 300", "500" => "N = 500"])
    unique_k = sort!(unique(simulation_results_df.k))
    n_names_rn = AOG.renamer([Symbol(k) => "N = $k" for k in unique_k])
    unique_μ = sort!(unique(simulation_results_df.μ))
    simulation_results_df.μ_int = Int.(simulation_results_df.μ)
    simulation_results_df.μ_sym = Symbol.(simulation_results_df.μ)
    unique_μ_int = sort!(unique(simulation_results_df.μ_int))
    μ_names_dict = Dict(μ => Printf.@sprintf("ρ = %.2f%%", μ) for μ in unique_μ)
    μ_names_rn = AOG.renamer([Symbol(μ) => Printf.@sprintf("ρ = %.2f%%", μ) for μ in unique_μ])
    simulation_results_df.μ_str = simulation_results_df.μ .|> x -> μ_names_dict[x]
    # n_names_rn = AOG.renamer([Symbol(50) => "N = 50", Symbol(100) => "N = 100", Symbol(300) => "N = 300", Symbol(500) => "N = 500"])

    simulation_results_df.k_sym = Symbol.(simulation_results_df.k)
    simulation_results_df.n_with_jitter = simulation_results_df.n .+ (rand(size(simulation_results_df, 1)) .- .5) .* 20

    vis = AOG.visual(AOG.Scatter, alpha = .8, markersize = 14)
    scales = AOG.scales(Marker = (; palette = ["Multilevel" => :circle, "Individual" => :utriangle, "Aggregated" => :xcross]))
    scales = AOG.scales(Marker = (; palette = [:circle, :utriangle, :xcross]))
    # scales = AOG.scales(Marker = (; palette = [:multilevel => :circle, :individual => :utriangle, :aggregated => :xcross]))

    w = 300
    d = AOG.data(subset(simulation_results_df, :k => x-> x .== 50))
    plt = d * AOG.mapping(:n => AOG.nonnumeric => "Time points (T)", :auc_indiv => "AUC", color = :method => method_names_rn => "", marker = :method => method_names_rn => "",#=row = :k_sym => n_names_rn => "N",=# col = :μ_sym => μ_names_rn => "ρ")
    fig = AOG.draw(plt * vis, scales,
        axis   = (xlabelrotation = pi/4, ),
        legend = (framevisible=false, ),
        figure = (size = (900, 300),))
    AOG.Label(fig.figure[0, 1:end-1], "Individual Level Edge Retrieval (N = 50)", tellwidth = false)
    fig

    AOG.save(joinpath(figures_dir, "individual_level_edge_retrieval_N_50.pdf"), fig)

    w = 300
    d = AOG.data(subset(simulation_results_df, :k => x-> x .== 50))

    plt = d * AOG.mapping(:n => AOG.nonnumeric => "Time points (T)", :auc_group => "AUC", color = :method => method_names_rn => "", marker = :method => method_names_rn => "", #=row = :k_sym => n_names_rn => "N",=# col = :μ_sym => μ_names_rn => "ρ")
    fig = AOG.draw(plt * vis, scales,
        axis   = (xlabelrotation = pi/4, ),
        legend = (framevisible=false, ),
        figure = (size = (900, 300),))
    AOG.Label(fig.figure[0, 1:end-1], "Group Level Edge Retrieval (N = 50)", tellwidth = false)

    AOG.save(joinpath(figures_dir, "group_level_edge_retrieval_N_50.pdf"), fig)

    # alternative view
    w = 300
    simulation_results_df.n_sym = Symbol.(simulation_results_df.n)
    μ_names_rn2 = AOG.renamer([Symbol(μ) => Printf.@sprintf("%.0f", μ) for μ in unique_μ])
    t_names_rn = AOG.renamer([Symbol(t) => "T = $t" for t in unique(simulation_results_df.n)])
    d = AOG.data(subset(simulation_results_df, :k => x-> x .== 50))
    plt = d * AOG.mapping(
        :μ_sym => μ_names_rn2 => "ρ",
        :auc_indiv => "AUC",
        color = :method => method_names_rn => "",
        marker = :method => method_names_rn => "",
        layout = :n_sym => t_names_rn => "Time points (T)"
    )
    fig = AOG.draw(plt * vis, scales,
        axis   = (xlabelrotation = pi/4, ),
        legend = (framevisible=false, ),
        figure = (size = w .* (2, 2),))
    AOG.Label(fig.figure[0, 1:2], "Individual Level Edge Retrieval (N = 50)", tellwidth = false)


    w = 150
    d = AOG.data(subset(simulation_results_df))
    plt = d * AOG.mapping(:n_with_jitter => "T", :auc_indiv => "AUC", color = :method => method_names_rn => "", marker = :method => method_names_rn => "", row = :k_sym => n_names_rn => "N", col = :μ_sym => μ_names_rn => "ρ")
    fig = AOG.draw(plt * vis, scales, legend = (framevisible=false, ), figure = (size = w .* (length(μ_names_dict), length(unique_k)),))
    AOG.Label(fig.figure[0, :], "Individual Level Edge Retrieval", tellwidth = false)
    fig
    AOG.save(joinpath(figures_dir, "individual_level_edge_retrieval.pdf"), fig)

    d = AOG.data(subset(simulation_results_df))
    plt = d * AOG.mapping(:n_with_jitter => "T", :auc_group => "AUC", color = :method => method_names_rn => "", marker = :method => method_names_rn => "", row = :k_sym => n_names_rn => "N", col = :μ => AOG.nonnumeric)
    fig = AOG.draw(plt * vis, scales, legend = (framevisible=false, ), figure = (size = w .* (length(μ_names_dict), length(unique_k)),))
    AOG.Label(fig.figure[0, :], "Group Level Edge Retrieval", tellwidth = false)
    fig
    AOG.save(joinpath(figures_dir, "group_level_edge_retrieval.pdf"), fig)

end

function main(
    ;
    results_dir = joinpath(pwd(), "simulation_study", "multilevel_vs_individual_vs_aggregated_results"),
    runs_dir    = joinpath(pwd(), "simulation_study", "multilevel_vs_individual_vs_aggregated_runs"),
    figures_dir = joinpath(pwd(), "simulation_study", "multilevel_vs_individual_vs_aggregated_figures"),
    overwrite::Bool = false,
    test_run::Bool = is_test_run())

    log_message("Starting Multilevel vs. Individual vs. Aggregated simulation study")

    if test_run
        !endswith("test", results_dir) && (results_dir *= "_test")
        !endswith("test", runs_dir   ) && (runs_dir    *= "_test")
        !endswith("test", figures_dir) && (figures_dir *= "_test")
    end

    !isdir(results_dir) && mkdir(results_dir)
    !isdir(runs_dir)    && mkdir(runs_dir)
    !isdir(figures_dir) && mkdir(figures_dir)

    simulation_results = run_simulation(results_dir, runs_dir, overwrite, test_run)
    save_figures(simulation_results, figures_dir)

    log_message("Finished Multilevel vs. Individual vs. Aggregated simulation study")

end

main()
