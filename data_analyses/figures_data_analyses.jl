using MultilevelGGMSampler
import StatsBase as SB, JLD2, CodecZlib, Printf, OnlineStats
using CairoMakie, LinearAlgebra
import ProgressMeter, Distributions, Random, DataFrames as DF, DelimitedFiles
import MLBase, Interpolations, LogExpFunctions
import Colors, ColorSchemes
import CSV
import OrderedCollections
import Graphs, NetworkLayout, GraphMakie

# optional for interactive plots
 import WGLMakie, JSServe, Bonito

#region functions

function get_posterior_means(obj_file, means_file)

    if isfile(means_file)
        return JLD2.load(means_file)["means"]
    else

        object = JLD2.jldopen(obj_file) do file

            # get_mean = first ∘ OnlineStats.value
            get_mean = x -> OnlineStats.mean(x.stats[1])

            samps = file["samples"]
            stats = samps.stats

            means_G = dropdims(SB.mean(samps.samples_G, dims = 4), dims = 4)
            means_K = get_mean.(stats.K)
            means_μ = get_mean.(stats.μ)
            means_σ = get_mean( stats.σ)

            (; means_G, means_K, means_μ, means_σ)
        end
        JLD2.jldsave(means_file, true; means = object)
        return object
        # return Dict{String, Any}("means" => object)
    end
end


function colorscheme_alpha(cscheme::ColorSchemes.ColorScheme, alpha::T = 0.5; ncolors=12) where T<:Real
    return ColorSchemes.ColorScheme([Colors.RGBA(ColorSchemes.get(cscheme, k), alpha) for k in range(0, 1, length=ncolors)])
end

function network_plot_2d(adj; kwargs...)
    f = Figure()
    ax = Axis(f[1, 1])
    network_plot_2d!(ax, adj; kwargs...)
    return f
end

function network_plot_2d!(ax, adj; kwargs...)
    # TODO: should be a weighted graph depending on the type of adj!
    g = Graphs.SimpleGraph(adj)
    GraphMakie.graphplot!(ax, g;
        kwargs...
    )
end

"""
Convert a color from RGB to RGBA with alpha value given by alpha.
"""
RGB_to_RGBA(rgb, alpha) = Colors.RGBA(rgb.r, rgb.g, rgb.b, alpha)

"""
Map a probability to an alpha value.

```julia
xx = -.1:.01:1.1
yy = prob_to_alpha_map.(xx)
lines(xx, yy)
```
"""
function prob_to_alpha_map(x; zero_below = 0.25, location = .5, scale = .1)
    dist = Distributions.truncated(Distributions.Logistic(location, scale), zero_below, 1)
    Distributions.cdf(dist, x)
end

"""
Compute the log inclusion Bayes factor from the posterior and prior inclusion probability
"""
function compute_log_incl_bf(post_incl_prob, prior_incl_prob)
    log_posterior_odds = LogExpFunctions.logit(post_incl_prob)
    log_prior_odds     = LogExpFunctions.logit(prior_incl_prob)
    return log_posterior_odds - log_prior_odds
end


function network_plot(obj, longnames_subnetworks, legend_elems, color_scheme, layout)

    fig = Figure(fontsize = color_scheme.fontsize, backgroundcolor = color_scheme.bg_col_fig)

    discrete_edges = obj isa AbstractArray{<:Integer} || obj isa Graphs.SimpleGraph{<:Integer}
    if discrete_edges
        # edge_color = color_scheme.edge_unweighted_color_alpha
        fig_idx    = 1
        leg_idx    = 1
    else
        # edge_color = color_scheme.edge_weighted_color_alpha
        colormap   = color_scheme.edge_weighted_color_alpha_scheme
        fig_idx    = 1:2
        leg_idx    = 1
        col_idx    = 2
    end


    ax = Axis(fig[fig_idx, 1], backgroundcolor = color_scheme.bg_col_ax)
    hidedecorations!(ax); hidespines!(ax)

    network_plot!(ax, obj, color_scheme; layout = layout)
    Legend(fig[leg_idx, 2], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
    if !discrete_edges
        Colorbar(fig[col_idx, 2], limits = (0, 1), colormap = color_scheme.edge_weighted_color_alpha_scheme, vertical = false, label = "Inclusion probability")
    end
    fig

end

function network_plot!(ax, adj, color_scheme;
    node_size  = 15,
    node_strokewidth = 1.5,
    node_attr  = (transparency = true, ),
    node_color = color_scheme.node_color,
    edge_color = :automatic,
    layout     = layout_fun2D,
    kwargs...
    )

    discrete_edges = adj isa AbstractArray{<:Integer} || adj isa Graphs.SimpleGraph{<:Integer}
    if edge_color === :automatic
        if discrete_edges
            edge_color = color_scheme.edge_unweighted_color_alpha
        else
            edge_color = color_scheme.edge_weighted_color_alpha
        end
    end

    g = Graphs.SimpleGraph(adj)
    GraphMakie.graphplot!(ax, g;
        node_size           = node_size,
        node_strokewidth    = node_strokewidth,
        node_attr           = node_attr,
        node_color          = node_color,
        edge_color          = edge_color,
        layout              = layout,
        kwargs...
    )

end

make_rgb(x) = Colors.RGB{Colors.N0f8}((x ./ 255)...)

function precision_to_partial(x, p = 116)
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

    # fpr_tpr = [reduce(vcat, fprs) ;; reduce(vcat, tprs)]

    # ci95_fprs = quantile(fprs, [0.25, 0.975])
    # ci95_tprs = quantile(tprs, [0.25, 0.975])

    fpr_catted = reduce(hcat, fprs)
    tpr_catted = reduce(hcat, tprs)

    h = 0.05
    ci = [h / 2, 1 - h / 2]
    ci_fpr = reduce(hcat, SB.quantile.(eachrow(fpr_catted), Ref(ci)))
    ci_tpr = reduce(hcat, SB.quantile.(eachrow(tpr_catted), Ref(ci)))

    xgrid = sort!(unique!(vcat(ci_fpr[1, :], ci_fpr[2, :])))

    itp_lower = Interpolations.interpolate((Interpolations.deduplicate_knots!(ci_fpr[2, :]),), ci_tpr[1, :], Interpolations.Gridded(Interpolations.Linear()))
    itp_upper = Interpolations.interpolate((Interpolations.deduplicate_knots!(ci_fpr[1, :]),), ci_tpr[2, :], Interpolations.Gridded(Interpolations.Linear()))

    lowerband = itp_lower[xgrid]
    upperband = itp_upper[xgrid]

    group_fpr = SB.mean(fprs)
    group_tpr = SB.mean(tprs)

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

function traceplots(samples::AbstractArray{T, 4}; kwargs...) where T

    k = size(samples, 3)
    p = size(samples, 1)

    if isone(k)
        nplots = MultilevelGGMSampler.n_edges(p)
        nr = isqrt(nplots)
        nc = ceil(Int, nplots / nr)
    else
        # nplots = MultilevelGGMSampler.n_edges(p) * k
        nr = p * (p + 1) ÷ 2# MultilevelGGMSampler.n_edges(p) + p
        nc = k
    end

    base_resolution = (300, 300)
    resolution = (nr, nc) .* base_resolution
    f = Figure(; size = resolution, kwargs...)

    ir, ic = 1, 1
    for ik in axes(samples, 3), ip in 1:size(samples, 1), jp in ip:size(samples, 2)
        ax = Axis(f[ir, ic], title = "i = $ip, j = $jp k = $ik")
        lines!(ax, samples[jp, ip, ik, :])
        ir += 1
        if ir > nr
            ir = 1
            ic += 1
        end
    end
    return f
end

function traceplots(samples::AbstractMatrix{T}; kwargs...) where T

    p_total = size(parent(samples), 1)
    p = size(samples, 1)

    nplots = p
    nr = isqrt(nplots)
    nc = ceil(Int, nplots / nr)

    base_resolution = (300, 300)
    resolution = (nr, nc) .* base_resolution
    f = Figure(; size = resolution, kwargs...)

    ir, ic = 1, 1
    for ip in axes(samples, 1)
        ii, jj = MultilevelGGMSampler.linear_index_to_triangle_index(ip, p_total)
        title = isone(length(axes(samples, 1))) ? "σ" : "i = $ii, j = $jj"
        ax = Axis(f[ir, ic], title = title)
        lines!(ax, samples[ip, :])
        ir += 1
        if ir > nr
            ir = 1
            ic += 1
        end
    end
    return f
end

dist_0(x) = sqrt(sum(abs2, x))

function scale_fun(pt, mx)
    d = dist_0(pt)
    d <= mx && return pt
    pt ./ 1.7
end

#endregion

function main(
    ;
    results_dir = joinpath(pwd(), "data_analyses", "fits"),
    figures_dir = joinpath(pwd(), "data_analyses", "figures"),
    test_run::Bool = is_test_run()
)

    test_run && !endswith(results_dir, "test") && (results_dir *= "_test")
    !isdir(results_dir) && error("Directory $results_dir does not exist. Run the multilevel analysis first.")

    test_run && !endswith(figures_dir, "test") && (figures_dir *= "_test")
    !isdir(figures_dir) && mkdir(figures_dir)

    files = (
        multilevel                    = joinpath(pwd(), results_dir, "multilevel.jld2"),
        individual                    = joinpath(pwd(), results_dir, "individual.jld2"),
        aggregated                    = joinpath(pwd(), results_dir, "aggregated.jld2"),
        multilevel_means              = joinpath(pwd(), results_dir, "multilevel_means.jld2"),
        multilevel_individual_samples = joinpath(pwd(), results_dir, "multilevel_individual_samples.jld2"),
        multilevel_group_samples      = joinpath(pwd(), results_dir, "multilevel_group_samples.jld2")
    )

    if !all(isfile, values(files))
        missing_files = filter(!isfile, values(files))

        error("""
The following results objects are missing:

$(join('\t' .* missing_files, "\n"))

Try rerunning `run_all_data_analyses.jl`.
""")

    end

    group_obj = JLD2.jldopen(files.multilevel_group_samples)

    # save the means to a file that can be used for simulations
    means_obj    = JLD2.load(files.multilevel_means)["means"]
    μ_if_present = SB.mean_and_std(μ for μ in means_obj.means_μ if μ > zero(μ))
    μ_if_absent  = SB.mean_and_var(μ for μ in means_obj.means_μ if μ <= zero(μ))

    JLD2.jldsave(joinpath(results_dir, "data_results_to_simulate_with.jld2"), true;
        μ_if_present           = μ_if_present,
        μ_if_absent            = μ_if_absent,
        group_graph_edge_probs = group_obj["graph_edge_probs"]
    )

    CairoMakie.activate!(inline = true)
#region setup for plot of group-level network

    atlas_root = joinpath(pwd(), "data_analyses", "atlas")
    txt_files = readdir(joinpath(atlas_root))
    filter!(endswith(".txt"), txt_files)

    col_names = replace.(txt_files, "Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T_" => "", ".txt" => "")

    info_df = DF.DataFrame(
        [
            col_names[i] => vec(DelimitedFiles.readdlm(joinpath(atlas_root, txt_files[i]), String))
            for i in 2:length(col_names)
        ]
    )

    longnames_subnetworks = [
        "Subcortical", # per comment Linda, used to be "Superior Colliculus",
        "Visual",
        "Somatomotor",
        "Dorsal Attention",
        "Ventral Attention",
        "Limbic",
        "Frontoparietal",
        "Default Mode Network"
    ]

    id_index_to_group = Int.(indexin(
        info_df.subnet_order_names, unique(info_df.subnet_order_names)
    ))

    positions = DelimitedFiles.readdlm(joinpath(atlas_root, txt_files[1]))

    centroids2D = positions[:, 1:2]

    layout2D = Makie.Point2.(eachrow(centroids2D))
    layout_fun2D(::Any) = layout2D

    coords_qgraph_file = joinpath(results_dir, "layout_fr_groupnetwork.csv")

    if !isfile(coords_qgraph_file)

        probs_file = joinpath(results_dir, "graph_edge_probs.csv")
        !isfile(probs_file) && CSV.write(probs_file, DF.DataFrame(group_obj["graph_edge_probs"], :auto))

        μs_file = joinpath(results_dir, "graph_mu_vals.csv")
        !isfile(μs_file) && CSV.write(μs_file, DF.DataFrame(MultilevelGGMSampler.tril_vec_to_sym(means_obj.means_μ, -1), :auto))

        # this step is done in R, see R/fruchterman_reingold.R
        rdir  = joinpath(pwd(), "data_analyses", "R")
        rfile = "fruchterman_reingold.R"
        cd(rdir) do
            run(`Rscript $rfile $results_dir`)
        end

    end

    coords_qgraph = Makie.Point2.(eachrow(DelimitedFiles.readdlm(coords_qgraph_file, ','; skipstart = 1)))
    layout2D_qgraph(::Any) = coords_qgraph

    # compute inclusion BFs
    log_incl_bfs = compute_log_incl_bf.(group_obj["edge_inclusion_probs"], group_obj["prior_edge_inclusion_probs"])

    bounds = [1 / 10, 1 / 3, 1, 3, 10]
    log_bounds = log.(bounds)

    egdes_above_3   = log_incl_bfs .>= log_bounds[4]
    egdes_below_3   = log_incl_bfs .<= log_bounds[2]
    egdes_between_3 = log_incl_bfs .>= log_bounds[2] .&& log_incl_bfs .<= log_bounds[4]
    sum(sum.((egdes_above_3, egdes_below_3, egdes_between_3)))

    adj_above_3   = MultilevelGGMSampler.tril_vec_to_sym(egdes_above_3,   -1)
    adj_below_3   = MultilevelGGMSampler.tril_vec_to_sym(egdes_below_3,   -1)
    adj_between_3 = MultilevelGGMSampler.tril_vec_to_sym(egdes_between_3, -1)

    graph_above_3 = Graphs.SimpleGraph(adj_above_3)

    layout_spring_above_3 = NetworkLayout.Spring(C=3.0, seed = 1)(adj_above_3)
    max_dist_0 = 5.5

    dists = dist_0.(layout_spring_above_3)
    idx_too_far = dists .> max_dist_0
    layout_spring_above_3_tweaked = scale_fun.(layout_spring_above_3, max_dist_0)


    group_colors = map(x->make_rgb(Colors.color_names[x]), unique(info_df.subnet_order_colors))
    group_colors_alpha = RGB_to_RGBA.(group_colors, .95)

    edge_color_map  = :plasma


    edge_color_scheme = ColorSchemes.diverging_rainbow_bgymr_45_85_c67_n256
    edge_color_alpha_scheme = ColorSchemes.ColorScheme([
        RGB_to_RGBA(get(edge_color_scheme, x), prob_to_alpha_map(x, zero_below = 0.5))
        # RGB_to_RGBA(get(edge_color_scheme, 0.5 + (1 - x) / 2), prob_to_alpha_map(x))
        for x in 0.0:0.01:1.0
    ])
    partial_cor_edge_color_alpha_scheme = ColorSchemes.ColorScheme([
        RGB_to_RGBA(get(edge_color_scheme, x), prob_to_alpha_map(x, zero_below = 0.1, location = .1))
        # RGB_to_RGBA(get(edge_color_scheme, 0.5 + (1 - x) / 2), prob_to_alpha_map(x))
        for x in 0.0:0.01:1.0
    ])



    edge_color       = get(edge_color_scheme,       group_obj["edge_inclusion_probs"])
    edge_color_alpha = get(edge_color_alpha_scheme, group_obj["edge_inclusion_probs"])


    node_color       = group_colors[id_index_to_group]
    node_color_alpha = RGB_to_RGBA.(node_color, .5)

    edge_color_gray        = parse(Colors.Colorant, "rgb(128, 128, 128)")
    edge_color_gray_alpha  = parse(Colors.Colorant, "rgba(128, 128, 128, .65)")
    edge_color_white       = parse(Colors.Colorant, "rgb(192, 192, 192)")
    edge_color_white_alpha = parse(Colors.Colorant, "rgba(192, 192, 192, .65)")

    # background color of figure and axes
    # bg_col_ax  = Colors.RGBA(1, 1, 1, .03)
    # bg_col_fig = Colors.RGBA(1, 1, 1, .45)
    bg_col_ax  = :white
    bg_col_fig = :white

    color_scheme_light = (
        fontsize                    = 24,
        group_colors_scheme              = ColorSchemes.ColorScheme(group_colors),

        edge_weighted_color              = edge_color,
        edge_weighted_color_alpha        = edge_color_alpha[.!iszero.(group_obj["edge_inclusion_probs"])],

        edge_weighted_color_scheme       = edge_color_scheme,
        edge_weighted_color_alpha_scheme = edge_color_alpha_scheme,

        edge_unweighted_color       = edge_color_gray,
        edge_unweighted_color_alpha = edge_color_gray_alpha,

        node_color                  = node_color,
        node_color_alpha            = node_color_alpha,

        bg_col_ax                   = bg_col_ax,
        bg_col_fig                  = bg_col_fig
    )

    color_scheme = color_scheme_light


    legend_elems = [
        MarkerElement(color = col, marker = Makie.Circle, markersize = 15)
        for col in group_colors
    ]
#endregion

#region compare posterior means to sample means

    obs_precs = JLD2.jldopen(files.multilevel) do obj

        obs_covs  = Array{Float64}(undef, obj["data"].p, obj["data"].p)#, obj["data"].k)
        obs_precs = Array{Float64}(undef, obj["data"].p, obj["data"].p, obj["data"].k)
        @views for ik in axes(obs_precs, 3)
            obs_covs .= obj["data"].sum_of_squares[:, :, ik] ./ obj["data"].n
            obs_precs[:, :, ik] = inv(Symmetric(obs_covs))
        end
        obs_precs
    end

    obs_k_vals = similar(means_obj.means_K)
    @views for ik in axes(obs_k_vals, 2)
        MultilevelGGMSampler.tril_to_vec!(obs_k_vals[:, ik], obs_precs[:, :, ik])
    end

    obs_k_vec   = vec(obs_k_vals)
    means_k_vec = vec(means_obj.means_K)

    fig_K_est_vs_K_obs = retrieval_plot(obs_k_vec, means_k_vec)
    save_figs && save(joinpath(figures_dir, "K_est_vs_K_obs.png"), fig_K_est_vs_K_obs)

    _, obs_partial_cors = precision_to_partial(obs_k_vals)
    _, est_partial_cors = precision_to_partial(means_obj.means_K)

    fig_partial_est_vs_partial_obs = retrieval_plot(vec(obs_partial_cors), vec(est_partial_cors))

    save(joinpath(figures_dir, "rho_est_vs_rho_obs.png"), fig_partial_est_vs_partial_obs)

#endregion

#region trace plots
    range_μ = 1:12
    JLD2.jldopen(files.multilevel) do obj

        groupSamples = obj["groupSamples"]
        fig_μ_trace_plots = traceplots(view(groupSamples.μ, range_μ, :))
        save(joinpath(figures_dir, "mu_trace_plots.pdf"), fig_μ_trace_plots)

        fig_σ_trace_plot = traceplots(reshape(groupSamples.σ, 1, :))
        save(joinpath(figures_dir, "sigma_trace_plot.pdf"), fig_σ_trace_plot)

    end

#endregion

#region inclusion BFs

    group_obj = JLD2.jldopen(files.multilevel_group_samples)
    # compute inclusion BFs
    log_incl_bfs = compute_log_incl_bf.(group_obj["edge_inclusion_probs"], group_obj["prior_edge_inclusion_probs"])

    bounds_text = ["1 / 10", "1 / 3", "1", "3", "10"]
    bounds      = [ 1 / 10,   1 / 3,   1,   3,   10]
    log_bounds = log.(bounds)

    bounds_prob_scale = bounds ./ (1 .+ bounds)
    # not exactly the same
    # sum((group_obj["edge_inclusion_probs"] .>= bounds_prob_scale[4]) .!= egdes_above_3)
    # sum((group_obj["edge_inclusion_probs"] .<= bounds_prob_scale[2]) .!= egdes_below_3)
    # sum((bounds_prob_scale[2] .<= group_obj["edge_inclusion_probs"] .< bounds_prob_scale[4]) .!= egdes_between_3)

    # TODO: rename to included, excluded and insufficient_evidence
    egdes_above_3   = log_incl_bfs .>= log_bounds[4]
    egdes_below_3   = log_incl_bfs .<= log_bounds[2]
    # egdes_between_3 = log_incl_bfs .>= log_bounds[2] .&& log_incl_bfs .<= log_bounds[4]
    egdes_between_3 = log_bounds[2] .<= log_incl_bfs .<= log_bounds[4]
    sum(sum.((egdes_above_3, egdes_below_3, egdes_between_3)))

    adj_above_3   = MultilevelGGMSampler.tril_vec_to_sym(egdes_above_3,   -1)
    adj_below_3   = MultilevelGGMSampler.tril_vec_to_sym(egdes_below_3,   -1)
    adj_between_3 = MultilevelGGMSampler.tril_vec_to_sym(egdes_between_3, -1)

    graph_above_3 = Graphs.SimpleGraph(adj_above_3)
    graph_below_3 = Graphs.SimpleGraph(adj_below_3)

    bounds_intensity = LogExpFunctions.logistic.(log_bounds)
    log_bounds_with_inf = [-Inf; log_bounds; Inf]
    bounds_cols = get(ColorSchemes.magma, bounds_intensity)

    # CairoMakie.activate!(inline = true)

    # hist! cannot handle infinities
    log_incl_bfs_no_inf = copy(log_incl_bfs)
    log_incl_bfs_no_inf[isinf.(log_incl_bfs)] .= maximum(filter(isfinite, log_incl_bfs)) + 1

    bounds_legend_elems = [
        LineElement(color = col, linestyle = nothing)
        for col in bounds_cols
    ]
    legend_elements = LineElement(color = :green, linestyle = nothing, points = Point2f[(0, 0), (0, 1), (1, 0), (1, 1)])
    fig = Figure()
    ax = Axis(fig[1, 1], ylabel = "Frequency", xlabel = "Log(Inclusion BF)")
    hist!(ax, log_incl_bfs_no_inf; color = :grey)
    vlines!(ax, log_bounds; color = bounds_cols)
    Legend(fig[1, 1],
        bounds_legend_elems, bounds_text, "Inclusion BF",
        halign = :right, valign = :top,
        tellheight = false,
        tellwidth = false, backgroundcolor = :transparent,
        framevisible = false
    )
    fig

    save(joinpath(figures_dir, "inclusion_BF_histogram.pdf"), fig)


    freq_log_incl_bf_by_category = [
        count(
            x -> log_bounds_with_inf[i] <= x <= log_bounds_with_inf[i+1],
            log_incl_bfs
        )
        for i in 1:length(log_bounds_with_inf) - 1
    ]
    freq_log_incl_bf_by_category[1] + freq_log_incl_bf_by_category[2]
    freq_log_incl_bf_by_category[3] + freq_log_incl_bf_by_category[4]
    freq_log_incl_bf_by_category[5] + freq_log_incl_bf_by_category[6]

    log_bounds_3 = log.([1 / 3, 3])
    log_bounds_3_with_inf = [-Inf; log_bounds_3; Inf]

    freq_log_incl_bf_by_category3 = [
        count(
            x -> log_bounds_3_with_inf[i] <= x <= log_bounds_3_with_inf[i+1],
            log_incl_bfs
        )
        for i in 1:length(log_bounds_3_with_inf) - 1
    ]
    freq_log_incl_bf_by_category3

    legend_elems = [
        MarkerElement(color = col, marker = Makie.Circle, markersize = 15)
        for col in group_colors
    ]

    fig = Figure(size = (2000, 900), fontsize = color_scheme.fontsize, backgroundcolor = color_scheme.bg_col_fig)
    ax = Axis(fig[1, 1], title = "Evidence for Inclusion\nInclusion BF > 3", backgroundcolor = color_scheme.bg_col_ax)
    hidedecorations!(ax); hidespines!(ax)
    network_plot!(ax, adj_above_3, color_scheme)

    ax = Axis(fig[1, 2], title = "Evidence for Exclusion\nInclusion BF < 1 / 3", backgroundcolor = color_scheme.bg_col_ax)
    hidedecorations!(ax); hidespines!(ax)
    network_plot!(ax, adj_below_3, color_scheme)

    ax = Axis(fig[1, 3], title = "Insufficient Evidence\n1 / 3 ≤ Inclusion BF ≤ 3", backgroundcolor = color_scheme.bg_col_ax)
    hidedecorations!(ax); hidespines!(ax)
    network_plot!(ax, adj_between_3, color_scheme)

    Legend(fig[1, 4], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
    fig

    save(joinpath(figures_dir, "group_brain_3_incl_BFs.pdf"), fig)

#endregion

#region plot group-network

    fig1 = network_plot(group_obj["graph_edge_probs"], longnames_subnetworks, legend_elems, color_scheme, layout2D)
    fig2 = network_plot(group_obj["graph_edge_probs"], longnames_subnetworks, legend_elems, color_scheme, layout2D_qgraph)
    fig3 = network_plot(graph_above_3, longnames_subnetworks, legend_elems, color_scheme, layout_fun2D)

    layout_spring_above_3 = NetworkLayout.Spring(C=3.0, seed = 1)(adj_above_3)
    max_dist_0 = 5.5

    dists = dist_0.(layout_spring_above_3)
    idx_too_far = dists .> max_dist_0
    layout_spring_above_3_tweaked = scale_fun.(layout_spring_above_3, max_dist_0)


    fig4 = network_plot(graph_above_3, longnames_subnetworks, legend_elems, color_scheme, x->layout_spring_above_3_tweaked)

    w = 600
    fig_5_only_2 = Figure(size = w .* (2, 1), fontsize = color_scheme.fontsize, backgroundcolor = color_scheme.bg_col_fig)

    network_area = fig_5_only_2[1, 1] = GridLayout()
    legend_area  = fig_5_only_2[1, 2] = GridLayout()


    ax = Axis(network_area[1, 1], backgroundcolor = color_scheme.bg_col_ax)
    hidedecorations!(ax); hidespines!(ax)
    network_plot!(ax, group_obj["graph_edge_probs"], color_scheme, layout = layout2D_qgraph,
        #=edge_color = edge_color_temp=#)

    ax = Axis(network_area[1, 2], backgroundcolor = color_scheme.bg_col_ax)
    hidedecorations!(ax); hidespines!(ax)
    network_plot!(ax, graph_above_3, color_scheme, layout = layout_spring_above_3_tweaked)

    Legend(legend_area[1, 1], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
    Colorbar(legend_area[2, 1], limits = (0, 1), colormap = color_scheme.edge_weighted_color_alpha_scheme, vertical = false, label = "Inclusion probability")


    colgap!(network_area, 60)
    colsize!(fig_5_only_2.layout, 1, Relative(2/3))
    rowsize!(legend_area, 2, Relative(1/3))

    fig_5_only_2

    save(joinpath(figures_dir, "group_network_2_panels.pdf"), fig_5_only_2)

    w = 600
    fig_5_only_3 = Figure(size = w .* (3, 1), fontsize = color_scheme.fontsize, backgroundcolor = color_scheme.bg_col_fig)

    network_area = fig_5_only_3[1, 1] = GridLayout()
    legend_area  = fig_5_only_3[1, 2] = GridLayout()


    ax = Axis(network_area[1, 1], backgroundcolor = color_scheme.bg_col_ax)
    hidedecorations!(ax); hidespines!(ax)
    network_plot!(ax, group_obj["graph_edge_probs"], color_scheme, layout = layout_spring_above_3_tweaked)
    Label(network_area[0, 1], "Inclusion probability", tellwidth = false)

    ax = Axis(network_area[1, 2], backgroundcolor = color_scheme.bg_col_ax)
    hidedecorations!(ax); hidespines!(ax)
    network_plot!(ax, graph_above_3, color_scheme, layout = layout_spring_above_3_tweaked)
    Label(network_area[0, 2], "Inclusion BF > 3", tellwidth = false)

    ax = Axis(network_area[1, 3], backgroundcolor = color_scheme.bg_col_ax)
    hidedecorations!(ax); hidespines!(ax)
    network_plot!(ax, graph_below_3, color_scheme, layout = layout_spring_above_3_tweaked)
    Label(network_area[0, 3], "Inclusion BF < 1/3", tellwidth = false)

    Legend(legend_area[1, 1], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
    # Colorbar(legend_area[2, 1], limits = (0.0, 1.0), colormap = color_scheme.edge_weighted_color_alpha_scheme, vertical = false, label = "Inclusion probability",
        # minortickcolor = :black, minorticks = [1/11, 10/11], minorticksvisible=true, minortickalign = 1.0, minorticksize=12) # same as default for `size`

    Colorbar(legend_area[2, 1], limits = (0.0, 1.0), colormap = color_scheme.edge_weighted_color_alpha_scheme, vertical = false, label = "Inclusion probability",
        # flipaxis = false,
        ticks = ([0.0, 1/11, .5, 10/11, 1.0], ["0.0", "\n\nBF<1/3\n｜", "0.5", "\n\nBF>3\n｜", "1.0"]),
        tickalign = 0.0,
        minortickcolor = :black, minorticks = [1/11, 10/11], minorticksvisible=true, minortickalign = 1.0, minorticksize=12
    )


    # Box(legend_area[1, 1], strokecolor = :blue)
    # Box(legend_area[2, 1], color = :transparent, strokecolor = :blue)

    colgap!(network_area, 80)
    colgap!(fig_5_only_3.layout, 120)
    colsize!(fig_5_only_3.layout, 1, Relative(2/3))
    rowsize!(legend_area, 2, Relative(1/3))

    fig_5_only_3
    save(joinpath(figures_dir, "group_network_3_panels.pdf"), fig_5_only_3)


#endregion

#region group level edge inclusion heatmap

    ord = sortperm(id_index_to_group)
    id_index_to_group_sorted = sort(id_index_to_group)
    # hm = MultilevelGGMSampler.tril_vec_to_sym(vec(SB.mean(means_obj.means_G, dims = 2)), -1)[ord, ord]
    hm = MultilevelGGMSampler.tril_vec_to_sym(group_obj["edge_inclusion_probs"], -1)[ord, ord]


    start_to_stop = [findfirst(==(i), id_index_to_group_sorted):findlast(==(i), id_index_to_group_sorted) for i in 1:8]

    # two figures for 2 setups for the ticks labels
    tickpositions = map(x -> (first(x) + last(x)) ÷ 2, start_to_stop)
    ticklabels = [rich(longnames_subnetworks[i], color = group_colors[i]) for i in eachindex(longnames_subnetworks)]
    ticks = (tickpositions, ticklabels)

    grid_pts1 = [
        (Point2f(-1.0, first(start_to_stop[i]) - 0.0), Point2f(117, first(start_to_stop[i]) - 0.0))
        for i in 1:8
    ]
    grid_pts1 = [grid_pts1; (Point2f(-1.0, last(last(start_to_stop)) + 1.0), Point2f(117, last(last(start_to_stop)) + 1.0))]
    grid_pts2 = [
        (Point2f(pt1[2], pt1[1]), Point2f(pt2[2], pt2[1]))
        for (pt1, pt2) in grid_pts1
    ]
    grid_pts = [grid_pts1; grid_pts2]

    fig = Figure()#figure_padding = (0, 7, 0, 7))
    xleft = -1
    ax = Axis(fig[1, 1], limits = (xleft, 116.5, xleft, 116.5), #=xautolimitmargin = (0.0f0, 30.0f0),=#
        xticks = ticks, yticks = ticks, xticklabelrotation = pi/7, yticklabelrotation = pi/7)
    hidedecorations!(ax, ticklabels=false)
    hidespines!(ax)
    hm_m = heatmap!(ax, 1:117, 1:117, hm)
    Colorbar(fig[1, 2], hm_m, label = "Inclusion probability", height = Makie.Relative(116 / (116.5-xleft)), tellheight = false, valign = :top)
    fig

    linesegments!(ax, grid_pts, color = Colors.GrayA(0.0, 1), linewidth = 1)

    fig

    save(joinpath(figures_dir, "group_network_area_heatmap.pdf"), fig)

    # setup 2 with manual tick labels
    fig = Figure(figure_padding = (0, 7, 0, 7))
    xleft = -43
    ax = Axis(fig[1, 1], limits = (xleft, 116.5, xleft, 116.5), xautolimitmargin = (0.0f0, 30.0f0))
    hidedecorations!(ax, ticklabels = false)
    hidespines!(ax)
    hm_m = heatmap!(ax, 1:117, 1:117, hm)
    Colorbar(fig[1, 2], hm_m, label = "Inclusion probability", height = Makie.Relative(116 / (116.5-xleft)), tellheight = false, valign = :top)
    fig
    pts = [
        (Point2f(1.0, first(start_to_stop[i]) - 0.0), Point2f(117.0, first(start_to_stop[i]) - 0.0))
        for i in 1:8
    ]
    pts2 = [
        (Point2f(pt1[2], pt1[1]), Point2f(pt2[2], pt2[1]))
        for (pt1, pt2) in pts
    ]
    linesegments!(ax, pts2, color = Colors.GrayA(0.0, 1), linewidth = 1)
    linesegments!(ax, pts, color = Colors.GrayA(0.0, 1), linewidth = 1)

    fig

    for i in 1:8
        p1 = Point2f(first(start_to_stop[i]), 0)
        p2 = Point2f(last(start_to_stop[i])+1, 0)
        p4 = Point2f(0, first(start_to_stop[i]))
        p3 = Point2f(0, last(start_to_stop[i])+1)
        bracket!(ax, p1, p2, text = [longnames_subnetworks[i]], orientation = :down, rotation = pi/7, align = (:right, :center), style = :curly, textcolor = group_colors[i])
        bracket!(ax, p3, p4, text = [longnames_subnetworks[i]], orientation = :down, rotation = pi/7, align = (:right, :center), style = :curly, textcolor = group_colors[i])
    end

    fig

    save(joinpath(figures_dir, "group_network_area_heatmap_manual_ticklabels.pdf"), fig)

#endregion

#region individual-level variance heatmap

    # using SB.var would subtract the mean, but that is already done by subtracting the group-level edge_inclusion probs
    # maximum(abs, SB.mean(means_obj.means_G .- group_obj["edge_inclusion_probs"], dims = 2))
    # ~0.078
    # using variance
    # hm_avg_heterogeneity = MultilevelGGMSampler.tril_vec_to_sym(SB.var.(eachrow(means_obj.means_G .- group_obj["edge_inclusion_probs"])), -1)[ord, ord]
    # variance using the group-level edge inclusion probabilities as means
    hm_avg_heterogeneity = MultilevelGGMSampler.tril_vec_to_sym(SB.mean.(abs2, eachrow(means_obj.means_G .- group_obj["edge_inclusion_probs"])), -1)[ord, ord]
    hm_avg_thresholded_heterogeneity = MultilevelGGMSampler.tril_vec_to_sym(SB.mean.(abs2, eachrow((means_obj.means_G .> .5) .- (group_obj["edge_inclusion_probs"] .> .5))), -1)[ord, ord]


    ord = sortperm(id_index_to_group)
    id_index_to_group_sorted = sort(id_index_to_group)
    start_to_stop = [findfirst(==(i), id_index_to_group_sorted):findlast(==(i), id_index_to_group_sorted) for i in 1:8]

    tickpositions = map(x -> (first(x) + last(x)) ÷ 2, start_to_stop)

    ticklabels = [rich(longnames_subnetworks[i], color = group_colors[i]) for i in eachindex(longnames_subnetworks)]
    ticks = (tickpositions, ticklabels)

    grid_pts1 = [
        (Point2f(-1.0, first(start_to_stop[i]) - 0.0), Point2f(117, first(start_to_stop[i]) - 0.0))
        for i in 1:8
    ]
    grid_pts1 = [grid_pts1; (Point2f(-1.0, last(last(start_to_stop)) + 1.0), Point2f(117, last(last(start_to_stop)) + 1.0))]
    grid_pts2 = [
        (Point2f(pt1[2], pt1[1]), Point2f(pt2[2], pt2[1]))
        for (pt1, pt2) in grid_pts1
    ]

    fig = Figure(figure_padding = (0, 7, 0, 7))
    xleft  = -1#-43
    xright = 117#6.5
    ax = Axis(fig[1, 1], limits = (xleft, xright, xleft, xright), xautolimitmargin = (0.0f0, 30.0f0),
        xticks = ticks, yticks = ticks, xticklabelrotation = pi/7, yticklabelrotation = pi/7)
    hidedecorations!(ax, ticklabels = false)
    hidespines!(ax)
    hm_m = heatmap!(ax, 1:117, 1:117, hm_avg_heterogeneity)
    Colorbar(fig[1, 2], hm_m, label = "Individual variance", height = Makie.Relative(116 / (116.5-xleft)), tellheight = false, valign = :top)

    linesegments!(ax, grid_pts1, color = Colors.GrayA(0.0, 1), linewidth = 1)
    linesegments!(ax, grid_pts2, color = Colors.GrayA(0.0, 1), linewidth = 1)

    fig

    save(joinpath(figures_dir, "individual_variance_heatmap_1panel.pdf"), fig)

    ax = Axis(fig[1, 3], limits = (xleft, xright, xleft, xright), xautolimitmargin = (0.0f0, 30.0f0),
        xticks = ticks, xticklabelrotation = pi/7, yticklabelrotation = pi/7)
    hidedecorations!(ax, ticklabels = false)
    hideydecorations!(ax, ticklabels = true)
    hidespines!(ax)
    hm_m = heatmap!(ax, 1:117, 1:117, hm_avg_thresholded_heterogeneity)
    Colorbar(fig[1, 4], hm_m, label = "Individual variance", height = Makie.Relative(116 / (116.5-xleft)), tellheight = false, valign = :top)
    linesegments!(ax, grid_pts1, color = Colors.GrayA(0.0, 1), linewidth = 1)
    linesegments!(ax, grid_pts2, color = Colors.GrayA(0.0, 1), linewidth = 1)

    Label(fig[0, 1:2], "Raw",         tellwidth = false)
    Label(fig[0, 3:4], "Thresholded", tellwidth = false)
    resize!(fig, 1000, 500)

    fig

    save(joinpath(figures_dir, "individual_variance_heatmap_2panel.pdf"), fig)

#endregion

#region comparison individual_analysis

    individual_results  = JLD2.jldopen(files.individual)
    group_info_obj      = JLD2.jldopen(files.multilevel_group_samples)

    group_thresholded_probs = group_info_obj["thresholded_probs"]

    individual_results_mean_incl_probs = vec(SB.mean(individual_results["samples"].samples_G, dims = 3))
    individual_results_mean_prec       = map(x -> SB.mean(x[1]), (individual_results["samples"].stats.K))
    _, individual_results_mean_parcor     = precision_to_partial(individual_results_mean_prec)
    individual_results_mean_parcor = vec(individual_results_mean_parcor)

    multilevel_results_mean_incl_probs = means_obj.means_G[:, 1]
    multilevel_results_mean_prec = means_obj.means_K[:, 1]
    _, multilevel_results_mean_parcor     = precision_to_partial(multilevel_results_mean_prec)
    multilevel_results_mean_parcor = vec(multilevel_results_mean_parcor)

    is_congruent_multilevel = (multilevel_results_mean_incl_probs .> .5) .== group_thresholded_probs
    is_congruent_individual = (individual_results_mean_incl_probs .> .5) .== group_thresholded_probs
    SB.mean(is_congruent_multilevel)
    SB.mean(is_congruent_individual)

    col_incongruent = RGB_to_RGBA(ColorSchemes.viridis[0.0], .5)
    col_congruent   = RGB_to_RGBA(ColorSchemes.viridis[1.0], .5)
    colors = ifelse.(is_congruent_multilevel, col_incongruent, col_congruent)

    SB.mean_and_var(individual_results_mean_incl_probs .* (1 .- individual_results_mean_incl_probs)) # (0.09693977383245876, 0.004904622459738414)
    SB.mean_and_var(multilevel_results_mean_incl_probs .* (1 .- multilevel_results_mean_incl_probs)) # (0.026400030050974518, 0.003228119548550006)
    println(Printf.@sprintf(
        "Average SE individual = %.4f\nAverage SE multilevel = %.4f",
        SB.mean(x->x*(1 - x), individual_results_mean_incl_probs),
        SB.mean(x->x*(1 - x), multilevel_results_mean_incl_probs)
    ))
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Standard Error Multilevel Analysis", ylabel = "Standard Error Individual Analysis")
    ablines!(ax, 0, 1, color = Colors.GrayA(0.5, .5))
    scatter!(ax,
        # multilevel_results_mean_incl_probs .* (1 .- multilevel_results_mean_incl_probs),
        # individual_results_mean_incl_probs .* (1 .- individual_results_mean_incl_probs),
        sqrt.(multilevel_results_mean_incl_probs .* (1 .- multilevel_results_mean_incl_probs)),
        sqrt.(individual_results_mean_incl_probs .* (1 .- individual_results_mean_incl_probs)),
        # color = colors
        color = Colors.GrayA(.1, .5)
    )
    fig
    save(joinpath(figures_dir, "SE_comparison.pdf"), fig)

    # 4 panel plot that is not very informative
    fig = Figure()
    ax11 = Axis(fig[1, 1])
    scatter!(ax11, individual_results_mean_parcor, individual_results_mean_incl_probs)
    ax21 = Axis(fig[2, 1])
    scatter!(ax21, multilevel_results_mean_parcor, multilevel_results_mean_incl_probs)
    ax12 = Axis(fig[1, 2])
    scatter!(ax12, individual_results_mean_parcor, multilevel_results_mean_parcor)
    ax22 = Axis(fig[2, 2])
    scatter!(ax22, multilevel_results_mean_incl_probs, individual_results_mean_incl_probs)
    fig



#endregion

#region roc curves

    expected_group_probs = group_obj["edge_inclusion_probs"]
    thresholded_individual_networks = means_obj.means_G .>= .5

    group_indiv_aucs = compute_auc_individual_group(thresholded_individual_networks, expected_group_probs)

    title_lhs = Printf.@sprintf("Individual ROCs")
    title_rhs = Printf.@sprintf("Average AUC: %.3f, 95%% CI: [%.3f, %.3f]",
        group_indiv_aucs.group.auc,
        MultilevelGGMSampler.compute_auc(group_indiv_aucs.group.ci_fpr[2, :], group_indiv_aucs.group.ci_tpr[1, :]),
        MultilevelGGMSampler.compute_auc(group_indiv_aucs.group.ci_fpr[1, :], group_indiv_aucs.group.ci_tpr[2, :])
    )

    fig = Figure(size = (1400, 900))
    ax = Axis(fig[1, 1]; title = title_lhs, ylabel = "True positive rate", xlabel = "False positive rate")
    ablines!(ax, 0, 1, color = :grey)
    for i in eachindex(group_indiv_aucs.individual.fpr, group_indiv_aucs.individual.tpr)
        lines!(ax, group_indiv_aucs.individual.fpr[i], group_indiv_aucs.individual.tpr[i])#,
                # color = age_value_colors[i])#meta_df3.age[i] <= age_bounds[1] ? :red : :blue)
    end

    fig

    ax2 = Axis(fig[1, 2]; title = title_rhs, ylabel = "True positive rate", xlabel = "False positive rate")
    band!(ax2, group_indiv_aucs.group.xgrid_band, group_indiv_aucs.group.ci_lowerband, group_indiv_aucs.group.ci_upperband; color = Colors.RGBA(0, 0, 255, .4))

    ablines!(ax2, 0, 1, color = :grey)
    lines!(ax2, group_indiv_aucs.group.fpr, group_indiv_aucs.group.tpr)

    fig

    save(joinpath(figures_dir, "individual_group_rocs_auc.pdf"), fig)

    limits = (-0.05, 1.05, -0.05, 1.05)
    fig = Figure(size = (500, 500))
    ax = Axis(fig[1, 1]; title = title_lhs, ylabel = "True positive rate", xlabel = "False positive rate")
    ablines!(ax, 0, 1, color = :grey)
    for i in eachindex(group_indiv_aucs.individual.fpr, group_indiv_aucs.individual.tpr)
        lines!(ax, group_indiv_aucs.individual.fpr[i], group_indiv_aucs.individual.tpr[i])
    end

    # get the box for the inset plot
    # based on https://discourse.julialang.org/t/cairomakie-inset-plot-at-specific-x-y-coordinates/84797/2
    bbox = lift(ax.scene.camera.projectionview, ax.scene.viewport) do _, pxa
        bl = Makie.project(ax.scene, Point2f(0.5, 0)) + pxa.origin
        tr = Makie.project(ax.scene, Point2f(1, 0.5)) + pxa.origin
        Rect2f(bl, tr - bl)
    end

    ax_inset = Axis(fig,
        bbox  = bbox,
        backgroundcolor=Colors.Gray(0.975),
        aspect = 1,
        title = "Group-level ROC",
        titlealign = :right
    )
    hidedecorations!(ax_inset)
    band!(ax_inset, group_indiv_aucs.group.xgrid_band, group_indiv_aucs.group.ci_lowerband, group_indiv_aucs.group.ci_upperband; color = Colors.RGBA(0, 0, 255, .4))
    ablines!(ax_inset, 0, 1, color = :grey)
    lines!(ax_inset, group_indiv_aucs.group.fpr, group_indiv_aucs.group.tpr)

    title_rhs2 = Printf.@sprintf("Average AUC: %.3f\n95%% CI: [%.3f, %.3f]",
        group_indiv_aucs.group.auc,
        MultilevelGGMSampler.compute_auc(group_indiv_aucs.group.ci_fpr[2, :], group_indiv_aucs.group.ci_tpr[1, :]),
        MultilevelGGMSampler.compute_auc(group_indiv_aucs.group.ci_fpr[1, :], group_indiv_aucs.group.ci_tpr[2, :])
    )
    text!(ax_inset, 1.0, 0.0, text = title_rhs2, fontsize = 14, align = (:right, :baseline), justification=:right)
    fig

    save(joinpath(figures_dir, "individual_group_rocs_auc_inset.pdf"), fig)


    # the same group probabilities but random networks with the same strength
    thresholded_individual_networks_random = zero(thresholded_individual_networks)
    no_included_edges = vec(sum(thresholded_individual_networks; dims = 1))
    for i in axes(thresholded_individual_networks, 2)
        idx = SB.sample(axes(thresholded_individual_networks, 1), no_included_edges[i]; replace = false)
        thresholded_individual_networks_random[idx, i] .= true
    end
    @assert vec(sum(thresholded_individual_networks_random; dims = 1)) == no_included_edges
    random_group_indiv_aucs = compute_auc_individual_group(thresholded_individual_networks_random, expected_group_probs)

    title_lhs = Printf.@sprintf("Individual ROCs")
    title_rhs = Printf.@sprintf("Average AUC: %.3f, 95%% CI: [%.3f, %.3f]",
        random_group_indiv_aucs.group.auc,
        MultilevelGGMSampler.compute_auc(random_group_indiv_aucs.group.ci_fpr[2, :], random_group_indiv_aucs.group.ci_tpr[1, :]),
        MultilevelGGMSampler.compute_auc(random_group_indiv_aucs.group.ci_fpr[1, :], random_group_indiv_aucs.group.ci_tpr[2, :])
    )


    fig = Figure(size = (500, 500))
    ax = Axis(fig[1, 1]; title = "Random Individual ROCs", ylabel = "True positive rate", xlabel = "False positive rate")
    ablines!(ax, 0, 1, color = :grey)
    for i in eachindex(random_group_indiv_aucs.individual.fpr, random_group_indiv_aucs.individual.tpr)
        lines!(ax, random_group_indiv_aucs.individual.fpr[i], random_group_indiv_aucs.individual.tpr[i])
    end

    fig
    save(joinpath(figures_dir, "random_individual_group_rocs_auc.pdf"), fig)

    # separate bundles
    degree_rel_group = vec(sum(means_obj.means_G .>= expected_group_probs; dims = 1))
    degree_rel_group_quantiles = SB.quantile(degree_rel_group, [.025, 0.975])
    idx_lower  = findall(degree_rel_group .<= degree_rel_group_quantiles[1])
    idx_higher = findall(degree_rel_group .>= degree_rel_group_quantiles[2])
    idx_lower_higher = [idx_lower; idx_higher]

    heatmap(means_obj.means_G[:, idx_lower_higher])

    group_indiv_aucs_sep = compute_auc_individual_group(thresholded_individual_networks[:, idx_lower_higher], expected_group_probs)

    limits = (-0.05, 1.05, -0.05, 1.05)
    fig = Figure(size = (500, 500))
    ax = Axis(fig[1, 1]; title = title_lhs, ylabel = "True positive rate", xlabel = "False positive rate")
    ablines!(ax, 0, 1, color = :grey)
    for i in eachindex(group_indiv_aucs_sep.individual.fpr, group_indiv_aucs_sep.individual.tpr)
        lines!(ax, group_indiv_aucs_sep.individual.fpr[i], group_indiv_aucs_sep.individual.tpr[i])
    end
    fig

    diffmat = Matrix{Float64}(undef, size(thresholded_individual_networks, 2), size(thresholded_individual_networks, 2))
    for i in axes(thresholded_individual_networks, 2), j in axes(thresholded_individual_networks, 2)
        diffmat[i, j] = SB.mean(thresholded_individual_networks[:, i] .!= thresholded_individual_networks[:, j])
    end
    heatmap(diffmat)

#endregion

#region interactive standalone group network plot

WGLMakie.activate!()
Bonito.Page(exportable=true, offline=true)
app = Bonito.App() do session

    f = Figure(size = 1000 .* (2.5, 1))
    ax = Axis(f[1:2, 1])
    hidedecorations!(ax); hidespines!(ax)
    pl1 = network_plot!(ax, group_obj["graph_edge_probs"], color_scheme, layout = layout2D_qgraph)

    ax = Axis(f[1:2, 2], backgroundcolor = color_scheme.bg_col_ax)
    hidedecorations!(ax); hidespines!(ax)
    pl2 = network_plot!(ax, graph_above_3, color_scheme, layout = layout_spring_above_3_tweaked)

    custom_info = info_df.region_names_short
    on_click_callback = Bonito.js"""(plot, index) => {
        console.log(plot)
        const {pos, color} = plot.geometry.attributes
        console.log(pos)
        const custom = $(custom_info)[index]
        return custom
    }
    """

    # ToolTip(figurelike, js_callback; plots=plots_you_want_to_hover)
    tooltip = WGLMakie.ToolTip(f, on_click_callback; plots = [pl1, pl2])

    Legend(f[1, 3], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
    Colorbar(f[2, 3], limits = (0, 1), colormap = color_scheme.edge_weighted_color_alpha_scheme, vertical = false, label = "Inclusion probability")

    return Bonito.DOM.div(f, tooltip)
end

Bonito.export_static(joinpath(figures_dir, "interactive_group-network.html"), app)

CairoMakie.activate!(inline = true)
#endregion

#region interactive standalone inclusion threshold

individual_included_threshold = .1
people_threshold = .5

individual_networks = means_obj.means_G .> individual_included_threshold
adj_mean = dropdims(SB.mean(individual_networks, dims = 2), dims = 2)
adj_thresh = adj_mean .> people_threshold
sum(adj_thresh)

# non-interactive version
fig = Figure()
ax = Axis(fig[1, 1])
hidedecorations!(ax); hidespines!(ax)
network_plot!(ax, MultilevelGGMSampler.tril_vec_to_sym(adj_thresh, -1), color_scheme, layout = layout2D_qgraph)
fig


WGLMakie.activate!(inline = false)
app = Bonito.App() do session

    index_slider = Bonito.Slider(range(0.0, 1.0, step = 0.05))

    index_slider2 = Bonito.Slider(range(0.0, 1.0, step = 0.05))
    # individual_included_threshold = .5
    people_threshold = .5

    # adj_thresh = map(index_slider, index_slider2) do (individual_included_threshold, people_threshold)
    #     # map(index_slider) do individual_included_threshold
    #         individual_networks = means_obj.means_G .> individual_included_threshold
    #         adj_mean = dropdims(SB.mean(individual_networks, dims = 2), dims = 2)
    #         Graphs.SimpleGraph(MultilevelGGMSampler.tril_vec_to_sym(adj_mean .> people_threshold, -1))
    #     # end
    # end
    adj_thresh = map((individual_included_threshold, people_threshold) -> begin
        # map(index_slider) do individual_included_threshold
            individual_networks = means_obj.means_G .> individual_included_threshold
            adj_mean = dropdims(SB.mean(individual_networks, dims = 2), dims = 2)
            Graphs.SimpleGraph(MultilevelGGMSampler.tril_vec_to_sym(adj_mean .> people_threshold, -1))
        # end
    end, index_slider, index_slider2)

    fig = Figure()
    ax = Axis(fig[1, 1])
    hidedecorations!(ax); hidespines!(ax)
    # network_plot!(ax, adj_thresh, color_scheme, layout = layout2D_qgraph,
    pl1 = GraphMakie.graphplot!(ax, adj_thresh;
        node_size           = 15,
        node_strokewidth    = 1.5,
        node_attr           = (transparency = true, ),
        node_color          = color_scheme.node_color,
        edge_color          = color_scheme.edge_unweighted_color_alpha,
        layout              = layout2D_qgraph
    )
    slider1 = Bonito.DOM.div("individual level threshold for an edge: ", index_slider, index_slider.value)
    slider2 = Bonito.DOM.div("% participants having an edge: ",          index_slider2, index_slider2.value)

    custom_info = info_df.region_names_short
    on_click_callback = Bonito.js"""(plot, index) => {
        console.log(plot)
        const {pos, color} = plot.geometry.attributes
        console.log(pos)
        const custom = $(custom_info)[index]
        return custom
    }
    """

    tooltip = WGLMakie.ToolTip(fig, on_click_callback; plots = pl1)

    # return JSServe.record_states(session, JSServe.DOM.div(tooltip, slider, fig))
    return Bonito.DOM.div(tooltip, Bonito.record_states(session, Bonito.DOM.div(slider1, slider2, fig)))

end

Bonito.export_static(joinpath(figures_dir, "interactive_proportion_having_edge2.html"), app)


# save_figs && open(joinpath(figdir, "interactive_proportion_having_edge.html"), "w") do io
#     println(io, """
#     <html>
#         <head>
#         </head>
#         <body>
#     """)
#     JSServe.Page(exportable=true, offline=true)
#     show(io, MIME"text/html"(), app)
#     println(io, """
#         </body>
#     </html>
#     """)
# end

CairoMakie.activate!(inline = true)

#endregion

end

import PyCall
igraph = PyCall.pyimport("igraph")
partial_correlations, _ = precision_to_partial(means_obj.means_K)

mean_edge_weights = MultilevelGGMSampler.tril_vec_to_sym(
    vec(SB.mean(partial_correlations, dims = 2))
)
hist(tril_to_vec(mean_edge_weights, -1))


adj_above_3_weighted = adj_above_3 .* mean_edge_weights
py_ig_adj_above_3_weighted = igraph.Graph.Weighted_Adjacency(adj_above_3_weighted, mode = "undirected")
py_ig_adj_above_3_unweighted = igraph.Graph.Weighted_Adjacency(adj_above_3,        mode = "undirected")

hist(filter(!iszero, tril_to_vec(adj_above_3_weighted, -1)))


# walktrap_weighted   = igraph.Graph.community_walktrap(py_ig_adj_above_3_weighted, steps = 4)
# walktrap_unweighted = igraph.Graph.community_walktrap(py_ig_adj_above_3_unweighted, steps = 4)
# membership_weighted = walktrap_weighted.as_clustering().membership
# membership_unweighted = walktrap_unweighted.as_clustering().membership

# membership_weighted = igraph.Graph.community_infomap(py_ig_adj_above_3_weighted).membership

# membership_weighted = igraph.Graph.community_edge_betweenness(py_ig_adj_above_3_weighted).as_clustering().membership

membership_weighted = igraph.Graph.community_edge_betweenness(py_ig_adj_above_3_weighted).as_clustering().membership

membership = membership_weighted#membership_unweighted

show(DF.DataFrame(
    groupname = longnames_subnetworks[id_index_to_group],
    wt_weighted_label  = membership::Vector{Int}#,
    # wt_unweighted_label  = membership_unweighted::Vector{Int}
), allrows = true)

adj_above_3_weighted_tril = MultilevelGGMSampler.tril_to_vec(adj_above_3_weighted, -1)
extremas = (-1, 1)#extrema(adj_above_3_weighted_tril)
normalizer(x, minn, maxx) = (x - minn) / (maxx - minn)
normalizer.(extremas, extremas...)

adj_above_3_weighted_tril_norm = normalizer.(adj_above_3_weighted_tril, extremas...)
edge_color_weighted = Colors.RGBA.(get(edge_color_scheme, adj_above_3_weighted_tril_norm), .!iszero.(adj_above_3_weighted_tril))

markers_labels = [
    (:circle, ":circle"),
    (:rect, ":rect"),
    (:diamond, ":diamond"),
    (:hexagon, ":hexagon"),
    (:cross, ":cross"),
    (:xcross, ":xcross"),
    (:utriangle, ":utriangle"),
    (:dtriangle, ":dtriangle"),
    (:ltriangle, ":ltriangle"),
    (:rtriangle, ":rtriangle"),
    (:pentagon, ":pentagon"),
    (:star4, ":star4"),
    (:star5, ":star5"),
    (:star6, ":star6"),
    (:star8, ":star8"),
    (:vline, ":vline"),
    (:hline, ":hline"),
    ('a', "'a'"),
    ('B', "'B'"),
    ('↑', "'\\uparrow'"),
    ('😄', "'\\:smile:'"),
    ('✈', "'\\:airplane:'"),
]

node_markers = map(first, markers_labels[membership .+ 1])
# node_markers = map(first, markers_labels[membership_unweighted .+ 1])


legend_elems_clustering = [
    MarkerElement(color = Colors.GrayA(.4, .8), marker = first(markers_labels[idx]), markersize = 15)
    for idx in 1:1+maximum(membership)
]
cluster_counts = SB.countmap(membership)
cluster_counts_str = [string(cluster_counts[idx]) for idx in 0:maximum(membership)]
# cluster_counts_str = fill("", length(legend_elems_clustering)) # hide the counts

map(first, markers_labels[membership .+ 1])

w = 600
fig = Figure(size = (w, w))
ax = Axis(fig[1, 1])#, title = "hoi")
hidedecorations!(ax); hidespines!(ax)
network_plot!(ax, group_obj["graph_edge_probs"], color_scheme,
    edge_color = edge_color_weighted,
    layout = layout_spring_above_3_tweaked,
    node_marker = node_markers
)

gl_legend = fig[1, 2] = GridLayout()
Legend(gl_legend[1, 1:2], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
Colorbar(gl_legend[2, 1], limits = (-1, 1), colormap = edge_color_scheme, vertical = true, label = "Partial correlation")
Legend(gl_legend[2, 2], legend_elems_clustering, cluster_counts_str, "Cluster Counts", framevisible = false, nbanks = 1)
colsize!(fig.layout, 1, Relative(3/5))
fig

save_figs && save(joinpath(figdir, "clustering_group_level.pdf"), fig)

# for teaching with transparent background
fig = Figure(size = (w, w), backgroundcolor=:transparent)
ax = Axis(fig[1, 1], backgroundcolor=:transparent)#, title = "hoi")
hidedecorations!(ax); hidespines!(ax)
network_plot!(ax, group_obj["graph_edge_probs"], color_scheme,
    edge_color = edge_color_weighted,
    layout = layout_spring_above_3_tweaked,
    node_marker = node_markers
)
gl_legend = fig[1, 2] = GridLayout()
Legend(gl_legend[1, 1], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
# Colorbar(gl_legend[2, 1], limits = (-1, 1), colormap = edge_color_scheme, vertical = true, label = "Partial correlation")
Legend(gl_legend[2, 1], legend_elems_clustering, cluster_counts_str, "Cluster Counts", framevisible = false, nbanks = 1)
# colsize!(fig.layout, 1, Relative(3/5))
fig

save_figs && save(joinpath(figdir, "clustering_group_level2.pdf"), fig)
save_figs && save(joinpath(figdir, "clustering_group_level2.png"), fig)



adj_above_3
adj_graph_above_3 = Matrix(Graphs.adjacency_matrix(graph_above_3, weights = mean_edge_weights, steps = 10))
graph_above_3



#endregion