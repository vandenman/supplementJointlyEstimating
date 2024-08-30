using MultilevelGGMSampler, CairoMakie, StatsBase
import Random, Interpolations, JLD2, CodecZlib
import Printf, MLBase, LinearAlgebra
import Colors, Dates
include("../utilities.jl")

function compute_auc_individual_group(invidivual_Gmat, expected_Gvec)

    tprs = [Vector{Float64}() for _ in axes(invidivual_Gmat, 2)]
    fprs = [Vector{Float64}() for _ in axes(invidivual_Gmat, 2)]
    aucs = Vector{Float64}(undef, size(invidivual_Gmat, 2))

    thresholds = 0.00:.001:1.00
    for i in axes(invidivual_Gmat, 2)
        truth = view(invidivual_Gmat, :, i)

        roc_values = MLBase.roc(truth, expected_Gvec, thresholds)
        tpr = reverse!(MLBase.true_positive_rate.(roc_values))
        fpr = reverse!(MLBase.false_positive_rate.(roc_values))

        dfpr = diff(fpr)
        auc = LinearAlgebra.dot(view(tpr, 1:length(tpr)-1), dfpr) + LinearAlgebra.dot(diff(tpr), dfpr) / 2

        tprs[i] = tpr
        fprs[i] = fpr
        aucs[i] = auc
    end

    fpr_catted = reduce(hcat, fprs)
    tpr_catted = reduce(hcat, tprs)

    h = 0.05
    ci = [h / 2, 1 - h / 2]
    ci_fpr = reduce(hcat, quantile.(eachrow(fpr_catted), Ref(ci)))
    ci_tpr = reduce(hcat, quantile.(eachrow(tpr_catted), Ref(ci)))

    xgrid = sort!(unique!(vcat(ci_fpr[1, :], ci_fpr[2, :])))

    # deduplicate_knots! to remove harmless warnings, about duplicate knots
    itp_lower = Interpolations.interpolate((Interpolations.deduplicate_knots!(ci_fpr[2, :]),), ci_tpr[1, :], Interpolations.Gridded(Interpolations.Linear()))
    itp_upper = Interpolations.interpolate((Interpolations.deduplicate_knots!(ci_fpr[1, :]),), ci_tpr[2, :], Interpolations.Gridded(Interpolations.Linear()))

    lowerband = itp_lower[xgrid]
    upperband = itp_upper[xgrid]

    group_fpr = mean(fprs)
    group_tpr = mean(tprs)

    return (
        individual = (fpr = fprs, tpr = tprs, auc = aucs),
        group      = (
            fpr = group_fpr,
            tpr = group_tpr,
            auc = MultilevelGGMSampler.compute_auc(group_fpr, group_tpr),
            ci_fpr = ci_fpr,
            ci_tpr = ci_tpr,
            ci_lowerband = lowerband,
            ci_upperband = upperband,
            xgrid_band   = xgrid
        )
    )
end

function run_simulation(results_dir, overwrite::Bool = false, test_run::Bool = true)

    data_generation_methods = (
        low_variance = function(n, p, k, n_iter, n_warmup)

            ne = MultilevelGGMSampler.p_to_ne(p)
            group_G_tril = rand(0:1, ne)
            group_G      = tril_vec_to_sym(group_G_tril, -1)

            μ_mean = 2.
            μs = (2 .* group_G_tril .- 1) .* μ_mean .+ randn(ne)

            group_structure = CurieWeissStructure(; μ = μs, σ = 0.1)

            rng = Random.default_rng()
            data, parameters = simulate_hierarchical_ggm(rng, n, p, k, GWishart(p, 3.0), group_structure)

            save_individual_precmats = false

            res = sample_MGGM(data, SpikeAndSlabStructure(σ_spike = 0.15, threaded = true), group_structure; n_iter = n_iter, n_warmup = n_warmup, save_individual_precmats = save_individual_precmats);

            posterior_means = extract_posterior_means(res)
            means_G = posterior_means.G
            means_μ = posterior_means.μ
            mean_σ  = posterior_means.σ

            expected_group_probs  = exp.(MultilevelGGMSampler.compute_log_marginal_probs_approx(CurieWeissDistribution(means_μ, mean_σ)))

            return (; data, parameters, group_G, group_structure, res, means_μ, mean_σ, expected_group_probs, means_G)
        end,

        high_variance = function(n, p, k, n_iter, n_warmup)

            ne = MultilevelGGMSampler.p_to_ne(p)
            group_G_tril = rand(0:1, ne)
            group_G      = tril_vec_to_sym(group_G_tril, -1)

            μ_mean = .4
            μs = (2 .* group_G_tril .- 1) .* μ_mean .+ randn(ne)

            group_structure = CurieWeissStructure(; μ = μs, σ = 0.1)

            rng = Random.default_rng()
            data, parameters = simulate_hierarchical_ggm(rng, n, p, k, GWishart(p, 3.0), group_structure)

            save_individual_precmats = false

            res = sample_MGGM(data, SpikeAndSlabStructure(σ_spike = 0.15, threaded = true), group_structure; n_iter = n_iter, n_warmup = n_warmup, save_individual_precmats = save_individual_precmats);

            posterior_means = extract_posterior_means(res)
            means_G = posterior_means.G
            means_μ = posterior_means.μ
            mean_σ  = posterior_means.σ

            expected_group_probs  = exp.(MultilevelGGMSampler.compute_log_marginal_probs_approx(CurieWeissDistribution(means_μ, mean_σ)))

            return (; data, parameters, group_G, group_structure, res, means_μ, mean_σ, expected_group_probs, means_G)

        end,

        mixture = function(n, p, k, n_iter, n_warmup)

            k1 = floor(Int, 0.8 * k)
            k2 = k - k1

            ne = MultilevelGGMSampler.p_to_ne(p)
            group_G_tril_1   = rand(0:1, ne)
            group_G1        = tril_vec_to_sym(group_G_tril_1, -1)

            n_different = floor(Int, .2 * ne)
            idx_different = sample(1:ne, n_different)

            group_G_tril_2 = copy(group_G_tril_1)
            group_G_tril_2[idx_different] .= 1 .- group_G_tril_1[idx_different]
            group_G2   = tril_vec_to_sym(group_G_tril_2, -1)

            μs1 = (2 .* group_G_tril_1 .- 1) .* 2.0 .+ randn(ne)
            μs2 = (2 .* group_G_tril_2 .- 1) .* 2.0 .+ randn(ne)

            group_structure_1 = CurieWeissStructure(; μ = μs1, σ = 0.1)
            group_structure_2 = CurieWeissStructure(; μ = μs2, σ = 0.1)

            rng = Random.default_rng()
            data0, parameters1 = simulate_hierarchical_ggm(rng, n, p, k1, GWishart(p, 3.0), group_structure_1)
            data1, parameters2 = simulate_hierarchical_ggm(rng, n, p, k2, GWishart(p, 3.0), group_structure_2)

            data = [data0 ;;; data1]

            parameters = MultilevelGGMSampler.CurieWeissParameters(
                [parameters1.K ;;; parameters2.K],
                [parameters1.G ;;; parameters2.G],
                # NOTE: these values do not (and cannot) make sense
                parameters1.μ,
                parameters1.σ
            )

            save_individual_precmats = false

            group_structure = CurieWeissStructure()
            res = sample_MGGM(data, SpikeAndSlabStructure(σ_spike = 0.15, threaded = true), group_structure; n_iter = n_iter, n_warmup = n_warmup, save_individual_precmats = save_individual_precmats);

            posterior_means = extract_posterior_means(res)
            means_G = posterior_means.G
            means_μ = posterior_means.μ
            mean_σ  = posterior_means.σ
            expected_group_probs  = exp.(MultilevelGGMSampler.compute_log_marginal_probs_approx(CurieWeissDistribution(means_μ, mean_σ)))

            return (; data, parameters, group_G1, group_G2, μs1, μs2, k1, k2, group_structure_1, group_structure_2, res, means_μ, mean_σ, expected_group_probs, means_G)

        end
    )

    n           = 1000
    p           = test_run ?  30 :     60
    k           = test_run ?  50 :    100
    n_iter      = test_run ? 200 : 10_000
    n_warmup    = test_run ? 100 :  2_000

    test_prefix = test_run ? "test_" : ""
    filename = joinpath(results_dir, "$(test_prefix)results.jld2")
    if !isfile(filename) || overwrite
        Random.seed!(42)
        simulation_result = NamedTuple(
            key => fun(n, p, k, n_iter, n_warmup)
            for (key, fun) in pairs(data_generation_methods)
        )
        log_message("Writing results to $filename")
        JLD2.jldsave(filename, true; simulation_result = simulation_result)
    else

        simulation_result = JLD2.jldopen(filename)["simulation_result"]

    end

    return simulation_result

end

function save_figures(simulation_results, figures_dir)

    auc_fig = Figure(size = (1400, 1400 / 3))

    titles = ("Low variance", "High variance", "Mixture")

    limits = (-0.05, 1.05, -0.05, 1.05)
    # (i, (key, sim_res)) = first(enumerate(pairs(simulation_results)))
    for (i, (key, sim_res)) in enumerate(pairs(simulation_results))

        grid = auc_fig[1, i] = GridLayout()

        means_G = sim_res.means_G
        thresholded_individual_networks = means_G .> .5
        expected_group_probs  = sim_res.expected_group_probs

        group_indiv_aucs = compute_auc_individual_group(thresholded_individual_networks, expected_group_probs)

        title_lhs = titles[i]
        title_rhs = Printf.@sprintf("AUC: %.3f", group_indiv_aucs.group.auc)

        ax_left = Axis(grid[1, 1]; title = title_lhs, ylabel = "True positive rate", xlabel = "False positive rate",
            aspect = 1, limits = limits)

        ablines!(ax_left, 0, 1, color = :grey)
        for i in eachindex(group_indiv_aucs.individual.fpr, group_indiv_aucs.individual.tpr)
            lines!(ax_left, group_indiv_aucs.individual.fpr[i], group_indiv_aucs.individual.tpr[i], alpha = .5)
        end

        # get the box for the inset plot
        # based on https://discourse.julialang.org/t/cairomakie-inset-plot-at-specific-x-y-coordinates/84797/2
        bbox = lift(ax_left.scene.camera.projectionview, ax_left.scene.viewport) do _, pxa
            bl = Makie.project(ax_left.scene, Point2f(0.5, 0))   + pxa.origin
            tr = Makie.project(ax_left.scene, Point2f(1, 0.5)) + pxa.origin
            Rect2f(bl, tr - bl)
        end


        ax_inset = Axis(auc_fig,
            bbox  = bbox,
            backgroundcolor=Colors.Gray(0.975),
            aspect = 1,
            title = "Group-level ROC",
            limits = limits,
            xautolimitmargin = (0.f0, 0.f0),
            yautolimitmargin = (0.f0, 0.f0),
            titlealign = :right
        )
        hidedecorations!(ax_inset)

        band!(ax_inset, group_indiv_aucs.group.xgrid_band, group_indiv_aucs.group.ci_lowerband, group_indiv_aucs.group.ci_upperband; color = Colors.RGBA(0, 0, 255, .4))
        ablines!(ax_inset, 0, 1, color = :grey)
        lines!(ax_inset, group_indiv_aucs.group.fpr, group_indiv_aucs.group.tpr)

        text!(ax_inset, 0.9, 0.0, text = title_rhs, fontsize = 14, align = (:right, :baseline))

    end

    save(joinpath(figures_dir, "roc_simulation.pdf"), auc_fig)

end

function main(;
    results_dir = joinpath(pwd(), "simulation_study", "roc_data"),
    figures_dir = joinpath(pwd(), "simulation_study", "roc_figures"),
    overwrite::Bool = false,
    test_run::Bool = is_test_run()
)

    log_message("Starting ROC simulation study")

    if test_run
        !endswith("test", results_dir) && (results_dir *= "_test")
        !endswith("test", figures_dir) && (figures_dir *= "_test")
    end

    !isdir(results_dir) && mkdir(results_dir)
    !isdir(figures_dir) && mkdir(figures_dir)

    simulation_results = run_simulation(results_dir, overwrite, test_run)
    save_figures(simulation_results, figures_dir)

    log_message("Finished ROC simulation study")

end

main()