using MultilevelGGMSampler
import StatsBase as SB, JLD2, CodecZlib, Printf, OnlineStats
using CairoMakie, LinearAlgebra
import WGLMakie, JSServe, Bonito
import ProgressMeter, Distributions, Random, DataFrames as DF, DelimitedFiles
import MLBase, Interpolations, LogExpFunctions
import Colors, ColorSchemes
import CSV
import OrderedCollections
CairoMakie.activate!(inline = true)
# include("../plotfunctions.jl")

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

    # TODO: remove warnings, they are harmless
    itp_lower = Interpolations.interpolate((ci_fpr[2, :],), ci_tpr[1, :], Interpolations.Gridded(Interpolations.Linear()))
    itp_upper = Interpolations.interpolate((ci_fpr[1, :],), ci_tpr[2, :], Interpolations.Gridded(Interpolations.Linear()))

    lowerband = itp_lower[xgrid]
    upperband = itp_upper[xgrid]

    group_fpr = SB.mean(fprs)
    group_tpr = SB.mean(tprs)

    return (
        individual = (fpr = fprs, tpr = tprs, auc = aucs),
        group      = (
            fpr = group_fpr,
            tpr = group_tpr,
            auc = compute_auc(group_fpr, group_tpr),
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


#endregion

save_figs = true
# root = joinpath(pwd(), "data_analyses", "fixed_model10")
root = joinpath("/home/don/hdd/surfdrive/Postdoc/ABC/simulations/", "data_analyses", "fixed_model10")
# root = joinpath(pwd(), "data_analyses", "age_split_fixed_model")


# obj_file = joinpath(root, "results_ss_new_cholesky_724_test=false_2.jld2")
obj_file = joinpath(root, "run.jld2")
@assert isfile(obj_file)

figdir = joinpath(root, "figures")
!isdir(figdir) && mkdir(figdir)

means_file = joinpath(root, "means_k_724_p_116_cholesky_2.jld2")
means_obj = get_posterior_means(obj_file, means_file)

group_obj = JLD2.jldopen(joinpath(root, "group_object_samples_k_724_p_116_with_prior_probs_cholesky.jld2"))


# μ_if_present = SB.mean_and_var(μ for μ in means_obj.means_μ if μ > zero(μ))
# μ_if_absent  = SB.mean_and_var(μ for μ in means_obj.means_μ if μ <= zero(μ))
μ_if_present = SB.mean_and_std(μ for μ in means_obj.means_μ if μ > zero(μ))
μ_if_absent  = SB.mean_and_var(μ for μ in means_obj.means_μ if μ <= zero(μ))
group_obj["graph_edge_probs"]

JLD2.jldsave(joinpath(pwd(), "simulation_study", "data_results_to_simulate_with.jld2"), true;
    μ_if_present           = μ_if_present,
    μ_if_absent            = μ_if_absent,
    group_graph_edge_probs = group_obj["graph_edge_probs"]
)

#region setup for plot of group-level network

atlas_root = joinpath(pwd(), "HCP-rfMRI-repository-main", "Atlases", "Schaefer", "100regions7networks")

txt_files = readdir(joinpath(atlas_root, "annotation"))
col_names = replace.(txt_files, "Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T_" => "", ".txt" => "")

info_df = DF.DataFrame(
    [
        col_names[i] => vec(DelimitedFiles.readdlm(joinpath(atlas_root, "annotation", txt_files[i]), String))
        for i in 2:length(col_names)
    ]
)
map(length ∘ unique, eachcol(info_df))
unique(info_df.subnet_order_names)
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

positions = DelimitedFiles.readdlm(joinpath(atlas_root, "annotation", txt_files[1]))

centroids2D = positions[:, 1:2]

layout2D = Makie.Point2.(eachrow(centroids2D))
layout_fun2D(::Any) = layout2D

coords_qgraph_file = joinpath(root, "layout_fr_groupnetwork.csv")
if !isfile(coords_qgraph_file)
    probs_file = joinpath(root, "graph_edge_probs.csv")
    !isfile(probs_file) && CSV.write(probs_file, DF.DataFrame(group_obj["graph_edge_probs"], :auto))
    μs_file = joinpath(root, "graph_mu_vals.csv")
    !isfile(μs_file) && CSV.write(μs_file, DF.DataFrame(MultilevelGGMSampler.tril_vec_to_sym(means_obj.means_μ, -1), :auto))

    # this step is done in R, see R/fruchterman_reingold.R


end
# TODO: remake these?
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
dist_0(x) = sqrt(sum(abs2, x))
dists = dist_0.(layout_spring_above_3)
idx_too_far = dists .> max_dist_0
function scale_fun(pt, mx)
    d = dist_0(pt)
    d <= mx && return pt
    pt ./ 1.7
end
layout_spring_above_3_tweaked = scale_fun.(layout_spring_above_3, max_dist_0)


group_colors = map(x->make_rgb(Colors.color_names[x]), unique(info_df.subnet_order_colors))
group_colors_alpha = RGB_to_RGBA.(group_colors, .95)

edge_color_map  = :plasma
# edge_color_scheme = getproperty(ColorSchemes, edge_color_map)
edge_color_scheme = ColorSchemes.diverging_rainbow_bgymr_45_85_c67_n256
# edge_color_scheme = ColorSchemes.ColorScheme([
#     ColorSchemes.plasma[x]
#     for x in 1.0:-0.01:0.5
# ])
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
    edge_weighted_color_alpha        = edge_color_alpha,

    edge_weighted_color_scheme       = edge_color_scheme,
    edge_weighted_color_alpha_scheme = edge_color_alpha_scheme,

    edge_unweighted_color       = edge_color_gray,
    edge_unweighted_color_alpha = edge_color_gray_alpha,

    node_color                  = node_color,
    node_color_alpha            = node_color_alpha,

    # left_mesh_colors            = left_mesh_colors,
    # right_mesh_colors           = right_mesh_colors,
    # left_mesh_colors_alpha      = left_mesh_colors_alpha,
    # right_mesh_colors_alpha     = right_mesh_colors_alpha,

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
obs_precs = JLD2.jldopen(obj_file) do obj

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
save_figs && save(joinpath(figdir, "K_est_vs_K_obs.png"), fig_K_est_vs_K_obs)

_, obs_partial_cors = precision_to_partial(obs_k_vals)
_, est_partial_cors = precision_to_partial(means_obj.means_K)

fig_partial_est_vs_partial_obs = retrieval_plot(vec(obs_partial_cors), vec(est_partial_cors))

save_figs && save(joinpath(figdir, "rho_est_vs_rho_obs.png"), fig_partial_est_vs_partial_obs)


#triangle plot

avg_est_partial_cors = vec(SB.mean(est_partial_cors, dims = 2))
avg_inclprobs        = vec(SB.mean(means_obj.means_G, dims = 2))

fig_triangle = Figure()
ax = Axis(fig_triangle[1, 1])
scatter!(ax, est_partial_cors[:, 1], means_obj.means_G[:, 1])
fig_triangle


# something with these two figures is clearly off!
fig_triangle = Figure()
ax = Axis(fig_triangle[1, 1])
scatter!(ax, avg_est_partial_cors, avg_inclprobs)
ax = Axis(fig_triangle[1, 2])
scatter!(ax, avg_est_partial_cors, means_obj.means_μ)
fig_triangle


means_obj.means_G
means_obj.means_K


#endregion

#region trace plots
# samples of K are in not stored due to memory concerns
# if !all(iszero, size(obj["samples"].samples_K))
#     range_p = 1:4
#     range_k = 1:5
#     fig_K_trace_plots = traceplots(view(obj["samples"].samples_K, range_p, range_p, range_k, :))
#     save(joinpath(figdir, "K_trace_plots.pdf"), fig_K_trace_plots)
# end

range_μ = 1:12
JLD2.jldopen(obj_file) do obj
    fig_μ_trace_plots = traceplots(view(obj["samples"].groupSamples.μ, range_μ, :))
    save_figs && save(joinpath(figdir, "μ_trace_plots.pdf"), fig_μ_trace_plots)

    fig_σ_trace_plot = traceplots(reshape(obj["samples"].groupSamples.σ, 1, :))
    save_figs && save(joinpath(figdir, "σ_trace_plot.pdf"), fig_σ_trace_plot)
end

# samples_μ = JLD2.jldopen(obj_file) do obj
#     obj["samples"].groupSamples.μ
# end
# samples_σ = JLD2.jldopen(obj_file) do obj
#     obj["samples"].groupSamples.σ
# end

# means_μ_more_burn = vec(SB.mean(view(samples_μ, :, 10_000:size(samples_μ, 2)), dims = 2))
# means_σ_more_burn = SB.mean(view(samples_σ, 10_000:length(samples_σ)))

# means_obj.means_μ
# fig = Figure()
# ax = Axis(fig[1, 1], xlabel = "less burnin", ylabel = "more burnin")
# ablines!(ax, 0, 1)
# scatter!(ax, means_obj.means_μ, means_μ_more_burn)
# fig

# means_obj_new0 = JLD2.jldopen("/home/don/hdd/surfdrive/Postdoc/ABC/simulations/data_analyses/fixed_model/means_k_724_p_116_cholesky_2.jld2")
# means_obj_new1 = JLD2.jldopen("/home/don/hdd/surfdrive/Postdoc/ABC/simulations/data_analyses/fixed_model2/means_k_724_p_116_cholesky_2.jld2")
# means_obj_new3 = JLD2.jldopen("/home/don/hdd/surfdrive/Postdoc/ABC/simulations/data_analyses/fixed_model3/means_k_724_p_116_cholesky_2.jld2")
# means_obj_new4 = JLD2.jldopen("/home/don/hdd/surfdrive/Postdoc/ABC/simulations/data_analyses/fixed_model4/means_k_724_p_116_cholesky_2.jld2")
# means_obj_old  = JLD2.jldopen("/home/don/hdd/surfdrive/Postdoc/ABC/simulations/data_analyses/means_k_724_p_116_cholesky_2.jld2")
# retrieval_plot(means_obj_old["means"].means_μ, means_obj.means_μ) # no constraint on μ
# retrieval_plot(means_obj_old["means"].means_μ, means_obj_new0["means"].means_μ) # 0.1
# retrieval_plot(means_obj_old["means"].means_μ, means_obj_new4["means"].means_μ) # 0.2
# retrieval_plot(means_obj_old["means"].means_μ, means_obj_new1["means"].means_μ) # 0.3
# retrieval_plot(means_obj_old["means"].means_μ, means_obj_new3["means"].means_μ) # 0.02

# retrieval_plot(means_obj_old["means"].means_μ, means_μ_more_burn)

# SB.mean(means_obj_old["means"].means_μ)
# SB.mean(means_obj.means_μ)

# rng = Random.default_rng()
# Random.seed!(rng, 1234)
# d = CurieWeissDistribution(means_μ_more_burn, means_σ_more_burn)
# s = Distributions.sampler(d)
# group_samples = Matrix{Int}(undef, length(d), 10_000); # should this be a BitArray? but then threading is impossible...
# prog = ProgressMeter.Progress(size(group_samples, 2), showspeed = true);
# Threads.@threads for ik in axes(group_samples, 2)
#     Distributions.rand!(rng, s, view(group_samples, :, ik))
#     ProgressMeter.next!(prog)
# end
# edge_inclusion_probs = vec(SB.mean(group_samples, dims = 2))
# graph_edge_probs        = MultilevelGGMSampler.tril_vec_to_sym(edge_inclusion_probs, -1)

# retrieval_plot(
#     edge_inclusion_probs,
#     group_obj["edge_inclusion_probs"]
# )

#endregion

#region inclusion BFs
group_obj = JLD2.jldopen(joinpath(root, "group_object_samples_k_724_p_116_with_prior_probs_cholesky.jld2"))
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
legend_elements = LineElement(color = :green, linestyle = nothing,
        points = Point2f[(0, 0), (0, 1), (1, 0), (1, 1)])
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


save_figs && save(joinpath(figdir, "inclusion_BF_histogram.pdf"), fig)


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

save_figs && save(joinpath(figdir, "group_brain_3_incl_BFs.pdf"), fig)

#endregion

#region plot group-network
fig1 = network_plot(group_obj["graph_edge_probs"], longnames_subnetworks, legend_elems, color_scheme, layout2D)
fig2 = network_plot(group_obj["graph_edge_probs"], longnames_subnetworks, legend_elems, color_scheme, layout2D_qgraph)
fig3 = network_plot(graph_above_3, longnames_subnetworks, legend_elems, color_scheme, layout_fun2D)

layout_spring_above_3 = NetworkLayout.Spring(C=3.0, seed = 1)(adj_above_3)
max_dist_0 = 5.5
dist_0(x) = sqrt(sum(abs2, x))
dists = dist_0.(layout_spring_above_3)
idx_too_far = dists .> max_dist_0
function scale_fun(pt, mx)
    d = dist_0(pt)
    d <= mx && return pt
    pt ./ 1.7
end
layout_spring_above_3_tweaked = scale_fun.(layout_spring_above_3, max_dist_0)
# layout_spring_above_3_tweaked = layout_spring_above_3

# put one unconnected node in a better spot
# unconnected_node = findall(v -> iszero(length(Graphs.neighbors(graph_above_3, v))), Graphs.vertices(graph_above_3))
# unconnected_node == 6
# layout_spring_above_3_tweaked[6]
# desired_distance = 0.95 * maximum(norm, view(layout_spring_above_3_tweaked, axes(layout_spring_above_3_tweaked, 1) .!= unconnected_node))
# layout_spring_above_3_tweaked[6] = Makie.Point2(desired_distance * sinpi(5/4), desired_distance * cospi(5/4))

# layout_spring_thresholded = reduce(vcat, collect.(layout_spring_above_3_tweaked)')
# CSV.write("data_analyses/results_ss_new_cholesky_724/layout_thesholded_groupnetwork.csv", DF.DataFrame(layout_spring_thresholded, :auto))

fig4 = network_plot(graph_above_3, longnames_subnetworks, legend_elems, color_scheme, x->layout_spring_above_3_tweaked)


# edge_color_alpha_scheme2 = ColorSchemes.ColorScheme([
#     RGB_to_RGBA(get(edge_color_scheme, x), prob_to_alpha_map(x, .5))
#     # RGB_to_RGBA(get(edge_color_scheme, 0.5 + (1 - x) / 2), prob_to_alpha_map(x))
#     for x in 0.0:0.01:1.0
# ])
# edge_color_temp = get(edge_color_alpha_scheme, group_obj["edge_inclusion_probs"])

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

# Box(legend_area[1, 1], strokecolor = :blue)
# Box(legend_area[2, 1], color = :transparent, strokecolor = :blue)

colgap!(network_area, 60)
colsize!(fig_5_only_2.layout, 1, Relative(2/3))
rowsize!(legend_area, 2, Relative(1/3))

fig_5_only_2

save_figs && save(joinpath(figdir, "group_network_2_panels.pdf"), fig_5_only_2)

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
save_figs && save(joinpath(figdir, "group_network_3_panels.pdf"), fig_5_only_3)


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

save_figs && save(joinpath(figdir, "group_network_area_heatmap.pdf"), fig)

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

# Colorbar(fig[1, 2], hm_m, label = "Inclusion probability")
fig
save_figs && save(joinpath(figdir, "group_network_area_heatmap_manual_ticklabels.pdf"), fig)

#endregion

#region comparison individual_analysis

# TODO: move these individual results to the correct location!
# individual_results  = JLD2.jldopen(joinpath(root, "individual_runs/run_k=1.jdl2"))
# individual_results  = JLD2.jldopen(joinpath(pwd(), "data_analyses", "fixed_model9", "individual_runs/run_k=1.jdl2"))
individual_results  = JLD2.jldopen(joinpath(root, "individual_runs", "run_k=1.jdl2"))

group_info_obj = JLD2.jldopen(joinpath(root, "group_object_samples_k_724_p_116_with_prior_probs_cholesky.jld2"))
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
save_figs && save(joinpath(figdir, "SE_comparison.pdf"), fig)

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

#region individual network_area

# individual_network_samples = JLD2.jldopen(joinpath(root, "G_samples_tril.jld2"))["indicator_samples_tril"]
info_df

# these could also be tuples?
idx_by_group = OrderedCollections.OrderedDict(
    nm => findfirst(==(nm), longnames_subnetworks)
    for nm in longnames_subnetworks
)
indices_by_group = OrderedCollections.OrderedDict(
    nm => findall(==(idx_by_group[nm]), id_index_to_group)
    for nm in longnames_subnetworks
)

dummy_mat = zeros(Int, 116, 116)
for nm in longnames_subnetworks
    indices = indices_by_group[nm]
    dummy_mat[indices, indices] .= idx_by_group[nm]
end
dummy_mat_tril = MultilevelGGMSampler.tril_to_vec(dummy_mat, -1)
egde_indices = OrderedCollections.OrderedDict(
    nm => findall(==(idx_by_group[nm]), dummy_mat_tril)
    for nm in longnames_subnetworks
)

function setup_dummy_mat(indices, idx)
    dummy_mat = zeros(Int, 116, 116)
    dummy_mat[indices, :] .= idx
    dummy_mat[:, indices] .= idx
    MultilevelGGMSampler.tril_to_vec(dummy_mat, -1)
end
egde_indices = OrderedCollections.OrderedDict(
    nm => findall(==(idx_by_group[nm]), setup_dummy_mat(indices_by_group[nm], idx_by_group[nm]))
    for nm in longnames_subnetworks
)

function setup_dummy_mat2(indices1, indices2)
    dummy_mat = zeros(Int, 116, 116)
    dummy_mat[indices1, indices2] .= 1
    dummy_mat[indices2, indices1] .= 1
    MultilevelGGMSampler.tril_to_vec(dummy_mat, -1)
end
edge_indices2 = Dict(
    (k1, k2) => findall(isone, setup_dummy_mat2(indices_by_group[k1], indices_by_group[k2]))
    for (k1, k2) in Iterators.product(keys(indices_by_group), keys(indices_by_group))
)
heterogeneity_mat = zeros(Float64, 8, 8)
i = 1
nmi = longnames_subnetworks[i]
j = 2
nmj = longnames_subnetworks[j]
for (i, nmi) in enumerate(longnames_subnetworks), (j, nmj) in enumerate(longnames_subnetworks)
    heterogeneity_mat[i, j] = SB.mean(SB.var.(eachrow(LogExpFunctions.logistic.(means_obj.means_G[edge_indices2[(nmi, nmj)], :] .- group_obj["edge_inclusion_probs"][edge_indices2[(nmi, nmj)]]))))
end
heatmap(heterogeneity_mat)

SB.mean(SB.var.(eachrow(means_obj.means_G[edge_indices2[(nmi, nmj)], :] .- group_obj["edge_inclusion_probs"][edge_indices2[(nmi, nmj)]])))

function isdiff2(x)
    # abs(x[1] - x[2]) > .3
    abs(x[1] - x[2]) > .3# && x[1] > .5
end

heterogeneity_mat2 = zeros(Float64, 8, 8)
heterogeneity_mat3 = zeros(Float64, 8, 8)
heterogeneity_mat4 = zeros(Float64, 8, 8)
for (i, nmi) in enumerate(longnames_subnetworks), (j, nmj) in enumerate(longnames_subnetworks)
    indices = edge_indices2[(nmi, nmj)]
    # heterogeneity_mat2[i, j] = count(isdiff2, zip(group_obj["edge_inclusion_probs"][indices], means_obj.means_G[indices]))
    heterogeneity_mat4[i, j] = SB.mean(isdiff2, zip(group_obj["edge_inclusion_probs"][indices], means_obj.means_G[indices]))
    # heterogeneity_mat4[i, j] = SB.mean(x->abs(x) > .25, vec(means_obj.means_G[indices, :] .- group_obj["edge_inclusion_probs"][indices]))
    heterogeneity_mat2[i, j] = SB.sum(SB.var.(eachrow(LogExpFunctions.logistic.(means_obj.means_G[indices, :] .- group_obj["edge_inclusion_probs"][indices]))))
    heterogeneity_mat3[i, j] = SB.mean(SB.var.(eachrow(LogExpFunctions.logistic.(means_obj.means_G[indices, :] .- group_obj["edge_inclusion_probs"][indices]))))
end
# heatmap(heterogeneity_mat4)

# Note: colorbar is not interpretable, just there to indicate that the two panels have a different scale?
ticklabels = [rich(longnames_subnetworks[i], color = group_colors[i]) for i in eachindex(longnames_subnetworks)]
ticks = (eachindex(longnames_subnetworks), ticklabels)
fig = Figure(size = (1000, 500))
ax = Axis(fig[1, 1], xticks = ticks, yticks = ticks, xticklabelrotation = pi/7, yticklabelrotation = pi/7)
hm_m = heatmap!(ax, heterogeneity_mat2)
Colorbar(fig[1, 2], hm_m)
ax = Axis(fig[1, 3], xticks = ticks, yticks = #= ticks =# (eachindex(longnames_subnetworks), fill("", length(longnames_subnetworks))), xticklabelrotation = pi/7)
hm_m = heatmap!(ax, heterogeneity_mat3)
Colorbar(fig[1, 4], hm_m)
Label(fig[0, 1:2], "Total variance",   tellwidth = false)
Label(fig[0, 3:4], "Average variance", tellwidth = false)
fig

save_figs && save(joinpath(figdir, "heterogeneity_heatmap.pdf"), fig)

isdiff(x) = isdiff(x[1], x[2])
isdiff(x, y) = (x < .5 && y > .5) || (x > .5 && y < .5)
OrderedCollections.OrderedDict
mismatch = NamedTuple(
    Symbol(nm) => count(isdiff, zip(group_obj["edge_inclusion_probs"][egde_indices[nm]], means_obj.means_G[egde_indices[nm]]))
    for nm in longnames_subnetworks
)

mean_variances = NamedTuple(
    Symbol(nm) => SB.mean(SB.var.(eachrow(means_obj.means_G[egde_indices[nm], :] .- group_obj["edge_inclusion_probs"][egde_indices[nm]])))
    for nm in longnames_subnetworks
)
total_variances = NamedTuple(
    Symbol(nm) => sum(SB.var.(eachrow(means_obj.means_G[egde_indices[nm], :] .- group_obj["edge_inclusion_probs"][egde_indices[nm]])))
    for nm in longnames_subnetworks
)



# using SB.var would subtract the mean, but that is already done by subtracting the group-level edge_inclusion probs
# maximum(abs, SB.mean(means_obj.means_G .- group_obj["edge_inclusion_probs"], dims = 2))
# ~0.078
# using variance
# hm_avg_heterogeneity = MultilevelGGMSampler.tril_vec_to_sym(SB.var.(eachrow(means_obj.means_G .- group_obj["edge_inclusion_probs"])), -1)[ord, ord]
# variance using the group-level edge inclusion probabilities as means
hm_avg_heterogeneity = MultilevelGGMSampler.tril_vec_to_sym(SB.mean.(abs2, eachrow(means_obj.means_G .- group_obj["edge_inclusion_probs"])), -1)[ord, ord]



ord = sortperm(id_index_to_group)
id_index_to_group_sorted = sort(id_index_to_group)
start_to_stop = [findfirst(==(i), id_index_to_group_sorted):findlast(==(i), id_index_to_group_sorted) for i in 1:8]


fig = Figure(figure_padding = (0, 7, 0, 7))
xleft = -43
ax = Axis(fig[1, 1], limits = (xleft, 116.5, xleft, 116.5), xautolimitmargin = (0.0f0, 30.0f0))
hidedecorations!(ax)
hidespines!(ax)
hm_m = heatmap!(ax, 1:117, 1:117, hm_avg_heterogeneity)
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

for i in 1:8
    p1 = Point2f(first(start_to_stop[i]), 0)
    p2 = Point2f(last(start_to_stop[i])+1, 0)
    p4 = Point2f(0, first(start_to_stop[i]))
    p3 = Point2f(0, last(start_to_stop[i])+1)
    bracket!(ax, p1, p2, text = [longnames_subnetworks[i]], orientation = :down, rotation = pi/7, align = (:right, :center), style = :curly, textcolor = group_colors[i])
    bracket!(ax, p3, p4, text = [longnames_subnetworks[i]], orientation = :down, rotation = pi/7, align = (:right, :center), style = :curly, textcolor = group_colors[i])
end

# Colorbar(fig[1, 2], hm_m, label = "Inclusion probability")
fig

# save_figs && save(joinpath(figdir, "group_network_area_heatmap.pdf"), fig)


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

save_figs && save(joinpath(figdir, "individual_variance_heatmap_1panel.pdf"), fig)

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

save_figs && save(joinpath(figdir, "individual_variance_heatmap_2panel.pdf"), fig)

#endregion

#region individual network_area

prior_incl_odds     = 1
posterior_incl_odds = means_obj.means_G ./ (1 .- means_obj.means_G)
inclusion_bf        = posterior_incl_odds ./ prior_incl_odds
included_edges = inclusion_bf .> 3

# this is only valid if we use the same Bayes factor thresholds!
# individual_included_threshold = .5
# included_edges = means_obj.means_G .> individual_included_threshold
included_edges_sum = vec(sum(included_edges, dims = 1))

median_included = SB.median(included_edges_sum)
participant_indices = (
    findmin(included_edges_sum)[2],
    findmax(included_edges_sum)[2],
    findmin(x-> abs(x - median_included), included_edges_sum)[2]
)

group_level_incl = sum(egdes_above_3)

sum(log_incl_bfs .> 0)
# group_level_incl = sum(group_obj["edge_inclusion_probs"] .> individual_included_threshold)

fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, included_edges_sum, title = "Density of included edges", xlabel = "Number of included edges", ylabel = "Frequency")
s1 = scatter!(ax, included_edges_sum[collect(participant_indices)], 0 .* included_edges_sum[collect(participant_indices)], color = :red)
s2 = scatter!(ax, group_level_incl, 0, color = :yellow)
axislegend(ax, [s1, s2], ["Individuals", "Group"])
fig

save_figs && save(joinpath(figdir, "individual_nws_density_histogram.pdf"), fig)


means_obj.means_G .> .95


w = 500
fig_5_only_2_one_participant = Figure(size = w .* (3, 2), fontsize = color_scheme.fontsize, backgroundcolor = color_scheme.bg_col_fig)

network_area = fig_5_only_2_one_participant[1, 1] = GridLayout()
legend_area  = fig_5_only_2_one_participant[1, 2] = GridLayout()

for (i, participant_idx) in enumerate(participant_indices)
# i, participant_idx = 1, 1

    included_G     = MultilevelGGMSampler.tril_vec_to_sym(included_edges[:, participant_idx], -1)
    pcor_vals, pcor_vals2 = precision_to_partial(means_obj.means_K[:, participant_idx])
    pcor_mat = MultilevelGGMSampler.tril_vec_to_sym(vec(pcor_vals))
    pcor_mat .*= included_G
    pcor_vals2 = pcor_vals2[included_edges[:, participant_idx]]

    edge_color_temp = get(partial_cor_edge_color_alpha_scheme, (pcor_vals2 .+ 1) ./ 2)

    ax = Axis(network_area[i, 1], backgroundcolor = color_scheme.bg_col_ax)
    hidedecorations!(ax); hidespines!(ax)
    network_plot!(ax, pcor_mat, color_scheme, layout = layout2D_qgraph,
        edge_color = edge_color_temp)

    ax = Axis(network_area[i, 2], backgroundcolor = color_scheme.bg_col_ax)
    hidedecorations!(ax); hidespines!(ax)
    network_plot!(ax, included_G, color_scheme, layout = layout_spring_above_3_tweaked)

end

Legend(legend_area[1, 1], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
Colorbar(legend_area[2, 1], limits = (-1, 1), colormap = partial_cor_edge_color_alpha_scheme, vertical = false, label = "Inclusion probability")

colsize!(fig_5_only_2_one_participant.layout, 1, Relative(3/4))
rowsize!(legend_area, 2, Relative(1/3))

fig_5_only_2_one_participant

save_figs && save(joinpath(figdir, "individual_network_2_panels.pdf"), fig_5_only_2_one_participant)


individual_network_samples = JLD2.jldopen(joinpath(root, "G_samples_tril.jld2"))["indicator_samples_tril"]

size(individual_network_samples)

p = MultilevelGGMSampler.ne_to_p(size(individual_network_samples, 1))
density_nodes_per_participant = zeros(Int, p, size(individual_network_samples, 2), size(individual_network_samples, 3))
size(density_nodes_per_participant)
prog = ProgressMeter.Progress(size(individual_network_samples, 2) * size(individual_network_samples, 3), showspeed = true)
Threads.@threads for i in axes(individual_network_samples, 2)
    @views for j in axes(individual_network_samples, 3)
        density_nodes_per_participant[:, i, j] = vec(sum(MultilevelGGMSampler.tril_vec_to_sym( individual_network_samples[:, i, j], -1), dims = 1))
        ProgressMeter.next!(prog)
    end
end

avg_density_nodes_per_participant = dropdims(SB.mean(density_nodes_per_participant, dims = 3), dims = 3)
var_avg_density_nodes_per_participant = SB.var.(eachrow(avg_density_nodes_per_participant))
median_var_avg_density_nodes_per_participant = SB.median(var_avg_density_nodes_per_participant)
idx_min = findmin(var_avg_density_nodes_per_participant)[2]
idx_med = findmin(x->abs(x - median_var_avg_density_nodes_per_participant), var_avg_density_nodes_per_participant)[2]
idx_max = findmax(var_avg_density_nodes_per_participant)[2]
idxes = (minimum = idx_min, median  = idx_med, maximum = idx_max)

fig = Figure()
for i in 1:3
    ax = Axis(fig[1, i], title = string(keys(idxes)[i]))
    hist!(ax, avg_density_nodes_per_participant[idxes[i], :])
end
Label(fig[0, :], "Included edges", tellwidth = false)
fig


qqs = SB.quantile(var_avg_density_nodes_per_participant, [0.1, 0.9])
heterogeneous_nodes = findall(>=(qqs[2]), var_avg_density_nodes_per_participant)
homogeneous_nodes   = findall(<=(qqs[1]), var_avg_density_nodes_per_participant)
other_nodes         = setdiff(1:116, [heterogeneous_nodes; homogeneous_nodes])

Dict{Int,Symbol}(i => (i in heterogeneous_nodes ? :heterogeneous : :homogeneous) for i in 1:116)


node_markers = [
    ifelse(i in heterogeneous_nodes, :rect,
        ifelse(i in homogeneous_nodes, :utriangle,
                :circle))
    for i in 1:116
]
node_size = 14 .+ 6 .* ((1:116) .∉(Ref(other_nodes)))

w = 600
fig_node_heterogeneity = Figure(size = w .* (2, 1), fontsize = color_scheme.fontsize, backgroundcolor = color_scheme.bg_col_fig)

network_area = fig_node_heterogeneity[1, 1] = GridLayout()
legend_area  = fig_node_heterogeneity[1, 2] = GridLayout()

ax = Axis(network_area[1, 1], backgroundcolor = color_scheme.bg_col_ax)
hidedecorations!(ax); hidespines!(ax)
network_plot!(ax, group_obj["graph_edge_probs"], color_scheme, layout = layout2D_qgraph,
    node_size = node_size, node_marker = node_markers)

ax = Axis(network_area[1, 2], backgroundcolor = color_scheme.bg_col_ax)
hidedecorations!(ax); hidespines!(ax)
network_plot!(ax, graph_above_3, color_scheme, layout = layout_spring_above_3_tweaked,
    node_size = node_size, node_marker = node_markers)

Legend(legend_area[1, 1], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
Colorbar(legend_area[2, 1], limits = (0, 1), colormap = color_scheme.edge_weighted_color_alpha_scheme, vertical = false, label = "Inclusion probability")

colsize!(fig_node_heterogeneity.layout, 1, Relative(3/4))
rowsize!(legend_area, 2, Relative(1/3))

fig_node_heterogeneity
save_figs && save(joinpath(figdir, "node_heterogeneity.pdf"), fig_node_heterogeneity)


var_density_nodes_per_participant = similar(avg_density_nodes_per_participant)

for i in axes(var_density_nodes_per_participant, 1), j in axes(var_density_nodes_per_participant, 2)
    var_density_nodes_per_participant[i, j] = SB.var(density_nodes_per_participant[i, j, :])
end
mapslices(SB.var, density_nodes_per_participant, dims = 2:3)
hist(avg_density_nodes_per_participant[1, :])
hist(avg_density_nodes_per_participant[2, :])

# This makes sense, but why does μ have an asymptote?
fig_mean_incl_prob_vs_μ = Figure()
ax = Axis(fig_mean_incl_prob_vs_μ[1, 1], xlabel = "Logit(mean inclusion probability)", ylabel = "Mean μ")
retrieval_plot!(ax,
    # LogExpFunctions.logit.(vec(SB.mean(means_obj.means_G, dims = 2))),
    # means_obj.means_μ
    vec(SB.mean(means_obj.means_G, dims = 2)),
    LogExpFunctions.logistic.(means_obj.means_μ)
)
ax = Axis(fig_mean_incl_prob_vs_μ[1, 2], xlabel = "mean individual inclusion probability", ylabel = "group inclusion prob")
retrieval_plot!(ax,
    vec(SB.mean(means_obj.means_G, dims = 2)),
    group_obj["edge_inclusion_probs"]
)
fig_mean_incl_prob_vs_μ

#endregion

#region interactive standalone group network plot

WGLMakie.activate!()
Bonito.Page(exportable=true, offline=true)
app = Bonito.App() do session
    # f, ax, pl = Makie.scatter(1:4, markersize=100, color=Float32[0.3, 0.4, 0.5, 0.6])
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

save_figs && Bonito.export_static(joinpath(figdir, "interactive_group-network.html"), app)

# save_figs && open(joinpath(figdir, "interactive_group-network.html"), "w") do io
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


# TODO: add explanatory text
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
Bonito.export_static(joinpath(figdir, "interactive_proportion_having_edge2.html"), app)


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

#region roc curves

# thresholded_individual_networks = MultilevelGGMSampler.graph_array_to_mat(means_obj.means_G .>= .5)

# expected_group_probs  = exp.(MultilevelGGMSampler.compute_log_marginal_probs_approx(CurieWeissDistribution(means_obj.means_μ, means_obj.means_σ)))
# # to validate both compute_log_marginal_probs_approx and the sampling based version
# retrieval_plot(group_obj["edge_inclusion_probs"], expected_group_probs)
# SB.meanad(group_obj["edge_inclusion_probs"], expected_group_probs)

#=
data_dir = joinpath("data_analyses", "data")
meta_df = CSV.read(joinpath(data_dir, "age.csv"), DF.DataFrame, skipto = 3)

DF.rename!(meta_df, 1 => :id, 2 => :age)

# DF.subset!(meta_df, :age => x-> x .!= 1200)
# age_bounds = SB.quantile(meta_df.age, [0.25, 0.75])
# count(<=(age_bounds[1]), meta_df.age) + count(>=(age_bounds[2]), meta_df.age)
# meta_df2 = DF.subset(meta_df, :age => x -> x .<= age_bounds[1] .|| x .>= age_bounds[2])

root_dir_files = "/home/don/hdd/surfdrive/Shared/GIN/"
@assert isdir(root_dir_files)
all_files = readdir(root_dir_files, join = true)
ids = map(x->x[1:10], basename.(all_files))

# all_files = all_files[ids .∈ Ref(meta_df2.id)]
# meta_df3 = meta_df2[indexin(meta_df2.id, getindex.(basename.(all_files), Ref(1:10))), :]
# @assert allunique(meta_df3.id)

meta_df3 = meta_df[filter(!isnothing, indexin(meta_df.id, getindex.(basename.(all_files), Ref(1:10)))), :]

age_value_01 = (meta_df3.age .- minimum(meta_df3.age)) ./ (maximum(meta_df3.age) - minimum(meta_df3.age))
age_value_colors = Colors.RGBA.(get(ColorSchemes.diverging_bkr_55_10_c35_n256, age_value_01), .5)
# ColorSchemes.diverging_bkr_55_10_c35_n256

fig = Figure(); ax = Axis(fig[1, 1], xlabel = "Age (months)", ylabel = "AUC")
leg = scatter!(ax,
    meta_df3.age,
    group_indiv_aucs.individual.auc,
    color = ifelse.(meta_df3.sex .== "M", :blue, :red)
)
fig

import GLM, StatsModels
meta_df3.auc = group_indiv_aucs.individual.auc
lm_fit = GLM.lm(StatsModels.@formula(auc ~ age * sex), meta_df3)
=#

expected_group_probs = group_obj["edge_inclusion_probs"]
thresholded_individual_networks = means_obj.means_G .>= .5

group_indiv_aucs = compute_auc_individual_group(thresholded_individual_networks, expected_group_probs)

title_lhs = Printf.@sprintf("Individual ROCs")
title_rhs = Printf.@sprintf("Average AUC: %.3f, 95%% CI: [%.3f, %.3f]",
    group_indiv_aucs.group.auc,
    compute_auc(group_indiv_aucs.group.ci_fpr[2, :], group_indiv_aucs.group.ci_tpr[1, :]),
    compute_auc(group_indiv_aucs.group.ci_fpr[1, :], group_indiv_aucs.group.ci_tpr[2, :])
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
save_figs && save(joinpath(figdir, "individual_group_rocs_auc.pdf"), fig)

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
    compute_auc(group_indiv_aucs.group.ci_fpr[2, :], group_indiv_aucs.group.ci_tpr[1, :]),
    compute_auc(group_indiv_aucs.group.ci_fpr[1, :], group_indiv_aucs.group.ci_tpr[2, :])
)
text!(ax_inset, 1.0, 0.0, text = title_rhs2, fontsize = 14, align = (:right, :baseline), justification=:right)
fig

save_figs && save(joinpath(figdir, "individual_group_rocs_auc_inset.pdf"), fig)


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
    compute_auc(random_group_indiv_aucs.group.ci_fpr[2, :], random_group_indiv_aucs.group.ci_tpr[1, :]),
    compute_auc(random_group_indiv_aucs.group.ci_fpr[1, :], random_group_indiv_aucs.group.ci_tpr[2, :])
)


fig = Figure(size = (500, 500))
ax = Axis(fig[1, 1]; title = "Random Individual ROCs", ylabel = "True positive rate", xlabel = "False positive rate")
ablines!(ax, 0, 1, color = :grey)
for i in eachindex(random_group_indiv_aucs.individual.fpr, random_group_indiv_aucs.individual.tpr)
    lines!(ax, random_group_indiv_aucs.individual.fpr[i], random_group_indiv_aucs.individual.tpr[i])
end

fig
save_figs && save(joinpath(figdir, "random_individual_group_rocs_auc.pdf"), fig)

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

import Clustering
clust_res = Clustering.kmeans(means_obj.means_G, 10)
SB.countmap(Clustering.assignments(clust_res))
idx7 = findall(==(7), Clustering.assignments(clust_res))
idx2 = findall(==(2), Clustering.assignments(clust_res))
idx9 = findall(==(9), Clustering.assignments(clust_res))
heatmap(diffmat[[idx2; idx7; idx9], [idx2; idx7; idx9]])
heatmap(diffmat[[idx7; idx9], [idx7; idx9]])
heatmap(diffmat[[idx2; idx9], [idx2; idx9]])


group_indiv_aucs_sep = compute_auc_individual_group(thresholded_individual_networks[:, [idx7; idx9]], expected_group_probs)

limits = (-0.05, 1.05, -0.05, 1.05)
fig = Figure(size = (500, 500))
ax = Axis(fig[1, 1]; title = title_lhs, ylabel = "True positive rate", xlabel = "False positive rate")
ablines!(ax, 0, 1, color = :grey)
for i in eachindex(group_indiv_aucs_sep.individual.fpr, group_indiv_aucs_sep.individual.tpr)
    lines!(ax, group_indiv_aucs_sep.individual.fpr[i], group_indiv_aucs_sep.individual.tpr[i])
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
    compute_auc(group_indiv_aucs.group.ci_fpr[2, :], group_indiv_aucs.group.ci_tpr[1, :]),
    compute_auc(group_indiv_aucs.group.ci_fpr[1, :], group_indiv_aucs.group.ci_tpr[2, :])
)
text!(ax_inset, 1.0, 0.0, text = title_rhs2, fontsize = 14, align = (:right, :baseline), justification=:right)
fig

#endregion

#region explained variance

function decompose_variances(indicator_samples)
    v(x) = begin
        n = length(x)
        s = sum(x)
        # to avoid floating point arithmetic until the last step, note that
        # μ * (1 - μ)
        # s / n * (1 - s / n)
        # μ / n * (n / n - μ / n)
        # 1 / n² * (s * (n - s))
        (s * (n - s)) / abs2(n)
    end
    total = v(vec(indicator_samples))
    comp1 = SB.var(vec(SB.mean(indicator_samples, dims = 2)), corrected = false)
    comp2 = SB.mean(v.(eachrow(indicator_samples)))
    # total = SB.var(vec(indicator_samples), corrected = false)
    # comp1 = SB.var(vec(SB.mean(indicator_samples, dims = 2)), corrected = false)
    # comp2 = SB.mean(vec(SB.var(indicator_samples, dims = 2, corrected = false)))

    total, comp1, comp2
end

function decompose_variances_mat(indicator_samples_tril)
    decompositions_data = Matrix{Float64}(undef, 3, size(indicator_samples_tril, 1))
    prog = ProgressMeter.Progress(size(decompositions_data, 2); showspeed = true)
    Threads.@threads for ie in axes(decompositions_data, 2)
        # mat = Int.(view(indicator_samples_tril, ie, :, :))
        mat = view(indicator_samples_tril, ie, :, :)
        decompositions_data[:, ie] .= decompose_variances(mat)
        ProgressMeter.next!(prog)
    end
    return decompositions_data
end


decompositions_data_file = joinpath(root, "variance_decomposition.jld2")
if !isfile(decompositions_data_file)

    indicator_samples_tril = JLD2.jldopen(joinpath(root, "G_samples_tril.jld2"))["indicator_samples_tril"]
    decompositions_data = decompose_variances_mat(indicator_samples_tril)
    JLD2.jldsave(decompositions_data_file, true; decompositions_data = decompositions_data)

else

    decompositions_data = JLD2.jldopen(decompositions_data_file)["decompositions_data"]

end


ord = sortperm(id_index_to_group)
id_index_to_group_sorted = sort(id_index_to_group)

explained_variances = decompositions_data[2, :] ./ decompositions_data[1, :]
explained_variances[isnan.(explained_variances)] .= 1.0

hm = MultilevelGGMSampler.tril_vec_to_sym(decompositions_data[3, :], -1)[ord, ord]


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
Colorbar(fig[1, 2], hm_m, label = "Explained Variance", height = Makie.Relative(116 / (116.5-xleft)), tellheight = false, valign = :top)
fig

linesegments!(ax, grid_pts, color = Colors.GrayA(0.0, 1), linewidth = 1)

fig

# save_figs && save(joinpath(figdir, "explained_variance_heatmap.pdf"), fig)


ord
heatmap(MultilevelGGMSampler.tril_vec_to_sym(explained_variances, -1))



decompositions_data_file = joinpath(root, "results_ss_new_cholesky_724", "variance_decomposition.jld2")
@assert isfile(joinpath(root, "results_ss_new_cholesky_724", "variance_decomposition.jld2"))

decompositions_obj = JLD2.jldopen(decompositions_data_file)
decompositions_data = decompositions_obj["decompositions_data"]

idx_nonzero = findall(!iszero, decompositions_data[1, :])
explained_variances = decompositions_data[2, idx_nonzero] ./ decompositions_data[1, idx_nonzero]
fig_explained_variance = Figure()
ax = Axis(fig_explained_variance[1, 1], ylabel = "Density", xlabel = "Explained variance")
density!(ax, explained_variances, color = Colors.GrayA(.3, .8))
fig_explained_variance

save_figs && save(joinpath(figdir, "density_explained_variance.pdf"), fig_explained_variance)

hist(explained_variances; bins = 50)

# the same for a bunch of random graphs
random_reps = JLD2.jldopen(joinpath(root, "results_ss_new_cholesky_724", "variance_decomposition_random_graphs.jld2"))["random_reps"]

@assert !any(isnan, random_reps)
@assert random_reps[1, :, :] ≈ random_reps[2, :, :] + random_reps[3, :, :]

any(iszero, random_reps[1, :, :])
random_explained_variances = random_reps[2, :, :] ./ random_reps[1, :, :]

fig_explained_variance_random_graphs = Figure()
ax = Axis(fig_explained_variance_random_graphs[1, 1], xlabel = "Explained Variance")
cols = Colors.distinguishable_colors(10)
RGB_to_RGBA(rgb, alpha) = Colors.RGBA(rgb.r, rgb.g, rgb.b, alpha)
cols = RGB_to_RGBA.(cols, .3)
for i in axes(random_explained_variances, 2)
    density!(ax, view(random_explained_variances, :, i),
        offset = -(i - 1) * 1000,
        color = cols[i], linestyle=:solid, strokewidth = 1, strokecolor = cols[i])
end
hideydecorations!(ax)
# note that the explained variance for random graphs are so small
# that it becomes impossible to show these on the random and data results on the same scale
fig_explained_variance_random_graphs
save_figs && save(joinpath(figdir, "density_explained_variance_random_graphs.pdf"), fig_explained_variance_random_graphs)


# do not remove zeros/ infinities
all_explained_variances = decompositions_data[2, :] ./ decompositions_data[1, :]

# set NaN to a small value, otherwise Graphs.SimpleGraph will remove the edge
# since we color the edges manually, that will then lead to a mismatch between the no. edges and no. colors
idx_isnan = findall(isnan, all_explained_variances)
all_explained_variances[isnan.(all_explained_variances)] .= 1e-100

explained_variances_sym_mat = MultilevelGGMSampler.tril_vec_to_sym(all_explained_variances, -1)

explained_variance_edge_cols   = get(edge_color_alpha_scheme, all_explained_variances)
unexplained_variance_edge_cols = get(edge_color_alpha_scheme, 1 .- all_explained_variances)
@assert length(explained_variance_edge_cols) == length(color_scheme.edge_weighted_color_alpha)

no_groups = length(unique(info_df.subnet_order_names))
no_group_counts = [count(==(i), id_index_to_group) for i in unique(id_index_to_group)]
centers = [2Point2f(sin(angle), cos(angle)) for angle in range(0, 2pi, no_groups + 1)] #
nested_circle_layout = Vector{Point2f}(undef, size(info_df, 1))
last_group_index = zero(no_group_counts)
stepsizes = 2pi ./ (no_group_counts)
for i in eachindex(nested_circle_layout)

    j = id_index_to_group[i]

    angle = last_group_index[j] * stepsizes[j]
    nested_circle_layout[i] = .5 * Point2f(sincos(angle)...) + centers[j]
    stepsizes

    last_group_index[j] += 1

end

scatter(nested_circle_layout, color = color_scheme.node_color)

nested_circle_layout_fun(::Any) = nested_circle_layout


fig_network_explained_variance = Figure()
ax = Axis(fig_network_explained_variance[1:2, 1])
hidedecorations!(ax); hidespines!(ax)
network_plot!(ax, explained_variances_sym_mat, color_scheme, layout = layout_spring_above_3_tweaked,
    edge_color = explained_variance_edge_cols
)
Legend(fig_network_explained_variance[1, 2], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
Colorbar(fig_network_explained_variance[2, 2], limits = (0, 1), colormap = edge_color_alpha_scheme, vertical = false, label = "Explained Variance")
fig_network_explained_variance

save_figs && save(joinpath(figdir, "network_explained_variance.pdf"), fig_network_explained_variance)

fig_network_unexplained_variance = Figure()
ax = Axis(fig_network_unexplained_variance[1:2, 1])
hidedecorations!(ax); hidespines!(ax)
network_plot!(ax, explained_variances_sym_mat, color_scheme, layout = layout_spring_above_3_tweaked,
    edge_color = unexplained_variance_edge_cols
)
Legend(fig_network_unexplained_variance[1, 2], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
Colorbar(fig_network_unexplained_variance[2, 2], limits = (0, 1), colormap = edge_color_alpha_scheme, vertical = false, label = "Unexplained Variance")
fig_network_unexplained_variance

save_figs && save(joinpath(figdir, "network_unexplained_variance.pdf"), fig_network_unexplained_variance)



#=
for quantile_cutoff in (.5, .625, .75, .9)
# quantile_cutoff = .75
    expl_variance_color_alpha_scheme = ColorSchemes.ColorScheme([
        RGB_to_RGBA(get(edge_color_scheme, x), prob_to_alpha_map(x; zero_below = SB.quantile(all_explained_variances, quantile_cutoff)))
        for x in 0.0:0.01:1.0
    ])
    unexpl_variance_color_alpha_scheme = ColorSchemes.ColorScheme([
        RGB_to_RGBA(get(edge_color_scheme, x), prob_to_alpha_map(x; zero_below = SB.quantile(1 .- all_explained_variances, quantile_cutoff)))
        for x in 0.0:0.01:1.0
    ])
    explained_variance_edge_cols   = get(expl_variance_color_alpha_scheme, all_explained_variances)
    unexplained_variance_edge_cols = get(unexpl_variance_color_alpha_scheme, 1 .- all_explained_variances)

    w = 500
    rel_size_legend = 2/5
    relative_width = 2 / (2 + rel_size_legend)
    figsize = (2 * w + rel_size_legend * w, 500) # 2 figures + legend
    fig_explained_variance_side_by_side = Figure(size = figsize)
    gl_plots  = fig_explained_variance_side_by_side[1, 1] = GridLayout()
    gl_legend = fig_explained_variance_side_by_side[1, 2] = GridLayout()
    ax_left   = Axis(gl_plots[1, 1])
    ax_middle = Axis(gl_plots[1, 2])
    ax_left.title   = "Explained variance"
    ax_middle.title = "Unexplained variance"
    colsize!(fig_explained_variance_side_by_side.layout, 1, Relative(relative_width))

    fig_explained_variance_side_by_side

    hidedecorations!(ax_left); hidespines!(ax_left)
    network_plot!(ax_left, explained_variances_sym_mat, color_scheme, layout = nested_circle_layout_fun,
        edge_color = explained_variance_edge_cols
    )

    hidedecorations!(ax_middle); hidespines!(ax_middle)
    network_plot!(ax_middle, explained_variances_sym_mat, color_scheme, layout = nested_circle_layout_fun,
        edge_color = unexplained_variance_edge_cols
    )

    Legend(gl_legend[1, 1], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
    Colorbar(gl_legend[2, 1], limits = (0, 1), colormap = expl_variance_color_alpha_scheme,   vertical = false, label = "Explained Variance")
    Colorbar(gl_legend[3, 1], limits = (0, 1), colormap = unexpl_variance_color_alpha_scheme, vertical = false, label = "Unexplained Variance")
    rowsize!(gl_legend, 1, Relative(2 / 3))
    # rowsize!(fig_explained_variance_side_by_side.layout[:, 2], 1, Relative(1 / 4))
    fig_explained_variance_side_by_side
    quantile_cutoff_str = Printf.@sprintf("%.3f", quantile_cutoff)
    # save_figs &&
    save(joinpath(figdir, "network_unexplained_variance_side_by_side_quantile_$quantile_cutoff_str.pdf"), fig_explained_variance_side_by_side)
end
=#

# quantile_cutoffs = (.5, .6, .7, .8, .9)
quantile_cutoffs = (.6, .7, .8, .9)
w = 500
rel_size_legend = 2/5
relative_width = 2 / (2 + rel_size_legend)
no_rows = length(quantile_cutoffs)
figsize = (2 * w + rel_size_legend * w, w * no_rows) # 2 figures + legend
fig_explained_variance_side_by_side_multiple_rows = Figure(size = figsize)
# gl_plots  = fig_explained_variance_side_by_side_multiple_rows[1, 1] = GridLayout()
# gl_legend = fig_explained_variance_side_by_side_multiple_rows[1, 2] = GridLayout()
# colsize!(fig_explained_variance_side_by_side_multiple_rows.layout, 1, Relative(relative_width))

for (i, quantile_cutoff) in enumerate(quantile_cutoffs)
    # quantile_cutoff = .5
    expl_variance_color_alpha_scheme = ColorSchemes.ColorScheme([
        RGB_to_RGBA(get(edge_color_scheme, x), prob_to_alpha_map(x; zero_below = SB.quantile(all_explained_variances, quantile_cutoff)))
        for x in 0.0:0.01:1.0
    ])
    # unexpl_variance_color_alpha_scheme = ColorSchemes.ColorScheme([
    #     RGB_to_RGBA(get(edge_color_scheme, x), prob_to_alpha_map(x; zero_below = SB.quantile(1 .- all_explained_variances, quantile_cutoff)))
    #     for x in 0.0:0.01:1.0
    # ])
    unexpl_variance_color_alpha_scheme = ColorSchemes.ColorScheme(reverse([
        RGB_to_RGBA(get(reverse(edge_color_scheme), x), prob_to_alpha_map(x; zero_below = SB.quantile(1 .- all_explained_variances, quantile_cutoff)))
        for x in 0.0:0.01:1.0
    ]))

    explained_variance_edge_cols   = get(expl_variance_color_alpha_scheme, all_explained_variances)
    # unexplained_variance_edge_cols = get(unexpl_variance_color_alpha_scheme, 1 .- all_explained_variances)
    unexplained_variance_edge_cols = get(unexpl_variance_color_alpha_scheme, all_explained_variances)

    unexplained_variance_edge_cols[idx_isnan] = explained_variance_edge_cols[idx_isnan]

    gl_plots  = fig_explained_variance_side_by_side_multiple_rows[i, 1] = GridLayout()
    # gl_legend = fig_explained_variance_side_by_side_multiple_rows[i, 2] = GridLayout()

    ax_left   = Axis(gl_plots[1, 1])
    ax_middle = Axis(gl_plots[1, 2])
    # ax_left   = Axis(gl_plots[i, 1])
    # ax_middle = Axis(gl_plots[i, 2])
    if i == 1
        ax_left.title   = "Explained variance"
        ax_middle.title = "Unexplained variance"
    end

    quantile_cutoff_str = Printf.@sprintf("%.3f", quantile_cutoff)
    ax_left.ylabel = "quantile: $quantile_cutoff_str"

    hidexdecorations!(ax_left); hideydecorations!(ax_left; label = false); hidespines!(ax_left)
    network_plot!(ax_left, explained_variances_sym_mat, color_scheme, layout = nested_circle_layout_fun,
        edge_color = explained_variance_edge_cols
    )

    hidedecorations!(ax_middle); hidespines!(ax_middle)
    network_plot!(ax_middle, explained_variances_sym_mat, color_scheme, layout = nested_circle_layout_fun,
        edge_color = unexplained_variance_edge_cols
    )

    # if i == 1
    #     Legend(gl_legend[1, 1], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
    # end

    # Colorbar(gl_legend[1, 1], limits = (0, 1), colormap = expl_variance_color_alpha_scheme,   vertical = false, label = "Explained Variance")
    # Colorbar(gl_legend[2, 1], limits = (0, 1), colormap = unexpl_variance_color_alpha_scheme, vertical = false, label = "Unexplained Variance")

    # rowsize!(gl_legend, 1, Relative(2 / 3))
        # rowsize!(fig_explained_variance_side_by_side.layout[:, 2], 1, Relative(1 / 4))
    # fig_explained_variance_side_by_side
    # quantile_cutoff_str = Printf.@sprintf("%.3f", quantile_cutoff)
        # save_figs &&
        # save(joinpath(figdir, "network_unexplained_variance_side_by_side_quantile_$quantile_cutoff_str.pdf"), fig_explained_variance_side_by_side)
end
legend_area = fig_explained_variance_side_by_side_multiple_rows[no_rows + 1, 1] = GridLayout()
colsize!(legend_area, 1, Relative(1/3))
Legend(legend_area[1:2, 1], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)

expl_variance_color_alpha_scheme   = ColorSchemes.ColorScheme(        [RGB_to_RGBA(get(edge_color_scheme, x), prob_to_alpha_map(x; zero_below = 0.0))          for x in 0.0:0.01:1.0])
# unexpl_variance_color_alpha_scheme = ColorSchemes.ColorScheme(        [RGB_to_RGBA(get(edge_color_scheme, x), prob_to_alpha_map(x; zero_below = 0.0))          for x in 0.0:0.01:1.0])
unexpl_variance_color_alpha_scheme = ColorSchemes.ColorScheme(reverse([RGB_to_RGBA(get(reverse(edge_color_scheme), x), prob_to_alpha_map(x; zero_below = 0.0)) for x in 0.0:0.01:1.0]))

fmax(f, x, y) = f(x) > f(y) ? x : y
joint_color_scheme = ColorSchemes.ColorScheme([
    fmax(Colors.alpha, expl_variance_color_alpha_scheme[x], unexpl_variance_color_alpha_scheme[x])
    for x in 0.0:0.01:1.0
])

Colorbar(legend_area[1:2, 2], limits = (0, 1), colormap = joint_color_scheme,   vertical = false, label = "Explained Variance")

# Colorbar(legend_area[1, 2], limits = (0, 1), colormap = expl_variance_color_alpha_scheme,   vertical = false, label = "Explained Variance")
# Colorbar(legend_area[2, 2], limits = (0, 1), colormap = unexpl_variance_color_alpha_scheme, vertical = false, label = "Unexplained Variance")
# rowsize!(fig_explained_variance_side_by_side_multiple_rows.layout, no_rows + 1, Relative(1.2 * (1 / (no_rows + 1))))
fig_explained_variance_side_by_side_multiple_rows

save_figs && save(joinpath(figdir, "network_unexplained_variance_side_by_side_multiple_quantiles.pdf"), fig_explained_variance_side_by_side_multiple_rows)

# SB.quantile(1 .- all_explained_variances, quantile_cutoff) ≈ 1 - SB.quantile(all_explained_variances, 1 - quantile_cutoff)

quantile_cutoff = .8
expl_variance_color_alpha_scheme = ColorSchemes.ColorScheme([
    RGB_to_RGBA(get(edge_color_scheme, x), prob_to_alpha_map(x; zero_below = SB.quantile(all_explained_variances, quantile_cutoff)))
    for x in 0.0:0.01:1.0
])
# unexpl_variance_color_alpha_scheme = ColorSchemes.ColorScheme([
#     RGB_to_RGBA(get(edge_color_scheme, x), prob_to_alpha_map(x; zero_below = SB.quantile(1 .- all_explained_variances, quantile_cutoff)))
#     for x in 0.0:0.01:1.0
# ])
unexpl_variance_color_alpha_scheme = ColorSchemes.ColorScheme(reverse([
    RGB_to_RGBA(get(reverse(edge_color_scheme), x), prob_to_alpha_map(x; zero_below = SB.quantile(1 .- all_explained_variances, quantile_cutoff)))
    for x in 0.0:0.01:1.0
]))
explained_variance_edge_cols   = get(expl_variance_color_alpha_scheme, all_explained_variances)
unexplained_variance_edge_cols = get(unexpl_variance_color_alpha_scheme, #=1 .-=# all_explained_variances)

unexplained_variance_edge_cols[idx_isnan] = explained_variance_edge_cols[idx_isnan]

lower_bound_quantile = SB.quantile(all_explained_variances, 1 - quantile_cutoff)
upper_bound_quantile = SB.quantile(all_explained_variances,     quantile_cutoff)
count(<=(lower_bound_quantile), all_explained_variances)
count(>=(upper_bound_quantile), all_explained_variances)

w = 500
rel_size_legend = 2/5
relative_width = 2 / (2 + rel_size_legend)
figsize = (2 * w + rel_size_legend * w, 500) # 2 figures + legend
fig_explained_variance_side_by_side = Figure(size = figsize)
gl_plots  = fig_explained_variance_side_by_side[1, 1] = GridLayout()
gl_legend = fig_explained_variance_side_by_side[1, 2] = GridLayout()
ax_left   = Axis(gl_plots[1, 1])
ax_middle = Axis(gl_plots[1, 2])
ax_left.title   = "Explained variance"
ax_middle.title = "Unexplained variance"
colsize!(fig_explained_variance_side_by_side.layout, 1, Relative(relative_width))

fig_explained_variance_side_by_side

hidedecorations!(ax_left); hidespines!(ax_left)
network_plot!(ax_left, explained_variances_sym_mat, color_scheme, layout = nested_circle_layout_fun,
    edge_color = explained_variance_edge_cols
)

hidedecorations!(ax_middle); hidespines!(ax_middle)
network_plot!(ax_middle, explained_variances_sym_mat, color_scheme, layout = nested_circle_layout_fun,
    edge_color = unexplained_variance_edge_cols
)

Legend(gl_legend[1, 1], legend_elems, longnames_subnetworks, framevisible = false, nbanks = 1)
Colorbar(gl_legend[2, 1], limits = (0, 1), colormap = expl_variance_color_alpha_scheme,   vertical = false, label = "Explained Variance")
Colorbar(gl_legend[3, 1], limits = (0, 1), colormap = unexpl_variance_color_alpha_scheme, vertical = false, label = "Unexplained Variance")
rowsize!(gl_legend, 1, Relative(2 / 3))

fig_explained_variance_side_by_side
save_figs && save(joinpath(figdir, "network_unexplained_variance_side_by_side_quantile_0.8.pdf"), fig_explained_variance_side_by_side)


hidedecorations!(ax_left); hidespines!(ax_left)
network_plot!(ax_left, explained_variances_sym_mat, color_scheme, layout = nested_circle_layout_fun,
    edge_color = explained_variance_edge_cols
)

# TODO: need to load thresholded_individual_networks
edge_variances = SB.var.(eachrow(thresholded_individual_networks))
minn, maxx = extrema(edge_variances)
edge_variances_rescaled = (edge_variances .- minn) ./ (maxx - minn)
edge_variances_rescaled[iszero.(edge_variances_rescaled)] .= 1e-10
extrema(edge_variances_rescaled)
edge_variances_nw = MultilevelGGMSampler.tril_vec_to_sym(edge_variances_rescaled, -1)

quantile_cutoff = .9

edge_variances_color_alpha_scheme = ColorSchemes.ColorScheme([
    RGB_to_RGBA(get(edge_color_scheme, x), prob_to_alpha_map(x; zero_below = SB.quantile(edge_variances_rescaled, quantile_cutoff)))
    for x in 0.0:0.01:1.0
])
edge_low_variances_color_alpha_scheme = ColorSchemes.ColorScheme(reverse([
    RGB_to_RGBA(get(reverse(edge_color_scheme), x), prob_to_alpha_map(x; zero_below = SB.quantile((1 - 1e-10) .- edge_variances_rescaled, quantile_cutoff)))
    for x in 0.0:0.01:1.0
]))

edge_variances_edge_cols   = get(edge_variances_color_alpha_scheme, edge_variances_rescaled)
edge_low_variances_edge_cols   = get(edge_low_variances_color_alpha_scheme, edge_variances_rescaled)

lower_bound_quantile = SB.quantile(edge_variances_rescaled, 1 - quantile_cutoff)
upper_bound_quantile = SB.quantile(edge_variances_rescaled,     quantile_cutoff)
count(<=(lower_bound_quantile), edge_variances_rescaled)
count(>=(upper_bound_quantile), edge_variances_rescaled)


fig = Figure()
ax_left = Axis(fig[1, 1]); hidedecorations!(ax_left); hidespines!(ax_left)
network_plot!(ax_left, edge_variances_nw, color_scheme, layout = nested_circle_layout_fun,
    edge_color = edge_variances_edge_cols
)
ax_middle = Axis(fig[1, 2]); hidedecorations!(ax_middle); hidespines!(ax_middle)
network_plot!(ax_middle, edge_variances_nw, color_scheme, layout = nested_circle_layout_fun,
    edge_color = edge_low_variances_edge_cols
)

fig

#endregion

#region degree distribution

group_info_obj = JLD2.jldopen(joinpath(root, "group_object_samples_k_724_p_116_with_prior_probs_cholesky.jld2"))
group_samples = group_info_obj["group_samples"]

group_samples_sym = BitArray(undef, 116, 116, size(group_samples, 2))
for i in axes(group_samples, 2)
    group_samples_sym[:, :, i] = tril_vec_to_sym(group_samples[:, i], -1)
end
degree_samples = dropdims(sum(group_samples_sym, dims = 2), dims = 2)

SB.percentile(degree_samples[:, 1], 80)
percentiles = [SB.percentile(col, 80) for col in eachcol(degree_samples)]
SB.mean(percentiles)

function compute_θ(adj, richness_percentile = 80)
    degree = dropdims(sum(adj, dims = 1); dims = 1)
    richness_cutoff = SB.percentile(degree, richness_percentile)
    rich_nodes_idx = findall(>=(richness_cutoff), degree)
    n_rich_nodes = length(rich_nodes_idx)
    subnet = @view adj[rich_nodes_idx, rich_nodes_idx]
    possible_edges = n_rich_nodes * (n_rich_nodes - 1) ÷ 2
    return (sum(subnet) ÷ 2) / possible_edges
end

function compute_clubness_values(group_samples_sym; richness_percentile_values = range(0, 100, length = 101))

    clubness_values = Matrix{Float64}(undef, size(group_samples_sym, 3), length(richness_percentile_values))

    n_random_reps = 100
    nv = size(group_samples_sym, 1)

    ne_max = nv * (nv - 1) ÷ 2

    unique_edge_counts = sort!(unique(vec(sum(group_samples_sym, dims = 1:2)) .÷ 2))
    # could use OrderedDict
    edge_count_to_idx = Dict(e=>i for (i, e) in enumerate(unique_edge_counts))

    # define here for non threaded version
    # mean_θ_rand = zeros(length(richness_percentile_values))
    # rand_adj_mat = zeros(Int, nv, nv)

    mean_θ_rand_by_degree = Matrix{Float64}(undef, length(richness_percentile_values), length(unique_edge_counts))

    prog = ProgressMeter.Progress(length(unique_edge_counts), showspeed = true, desc="Computing θ_rand: ");
    Threads.@threads for (i, ne) in collect(enumerate(unique_edge_counts))

        mean_θ_rand = zeros(length(richness_percentile_values))
        # fill!(mean_θ_rand, 0.0)
        linear_edge_indices = similar(1:ne_max, ne)

        rand_adj_mat = zeros(Int, nv, nv)
        for _ in 1:n_random_reps

            fill!(rand_adj_mat, 0)

            SB.sample!(1:ne_max, linear_edge_indices; replace = false)
            for idx in linear_edge_indices
                ii, jj = MultilevelGGMSampler.linear_index_to_lower_triangle_indices(idx, nv)
                rand_adj_mat[ii, jj] = rand_adj_mat[jj, ii] = 1
            end

            for (j, richness_percentile) in enumerate(richness_percentile_values)
                mean_θ_rand[j] += compute_θ(rand_adj_mat, richness_percentile)
            end

        end
        mean_θ_rand_by_degree[:, i] .= mean_θ_rand ./ n_random_reps
        ProgressMeter.next!(prog)
    end

    # could flips the loops and parallelize the outer one?
    prog = ProgressMeter.Progress(size(group_samples_sym, 3), showspeed = true, desc="Computing θ     : ");
    Threads.@threads for i in axes(group_samples_sym, 3)
        one_sample = @view group_samples_sym[:, :, i]
        for (j, richness_percentile) in enumerate(richness_percentile_values)
            θ_obs = compute_θ(one_sample, richness_percentile)
            ne = sum(one_sample) ÷ 2
            k = edge_count_to_idx[ne]
            mean_θ_rand = mean_θ_rand_by_degree[j, k]
            clubness_values[i, j] = θ_obs / mean_θ_rand
        end
        ProgressMeter.next!(prog)
    end
    return clubness_values
end

richness_percentile_values = range(0, 100, 1001)
clubness_values = compute_clubness_values(group_samples_sym; richness_percentile_values = richness_percentile_values)

clubness_cri = Matrix{Float64}(undef, 2, size(clubness_values, 2))
@views for i in axes(clubness_cri, 2)
    not_nan_idx = findall(!isnan, clubness_values[:, i])
    if length(not_nan_idx) < 3
        clubness_cri[:, i] = [NaN, NaN]
    else
        clubness_cri[:, i] .= SB.quantile(clubness_values[not_nan_idx, i], [.025, .975])
    end
end

# clubness_cri  = [SB.quantile(col, [.025, .975]) for col in eachcol(clubness_values)]
clubness_mean = SB.mean.(eachcol(clubness_values))

not_nan_idx = intersect(
    findall(!isnan, clubness_values[1, :]),
    findall(!isnan, clubness_values[2, :])
)

clubness_figure = Figure()
ax = Axis(clubness_figure[1, 1], xlabel = "Percentile cutoff for the club", ylabel = "Clubness")
band!(richness_percentile_values[not_nan_idx], clubness_cri[1, not_nan_idx], clubness_cri[2, not_nan_idx],
    color = Colors.GrayA(.1, .4))
lines!(richness_percentile_values, clubness_mean; color = :black)
clubness_figure

save_figs && save(joinpath(figdir, "rich_clubness.pdf"), clubness_figure)

#=
    mimics Figure 2 panel a of
    Bertolero, M. A., Yeo, B. T., & D’Esposito, M. (2017). The diverse club. Nature communications, 8(1), 1277.
    albeit that it's difficult to compare to their line for the "Rich club" due to the axes scales.
=#


lines(range(0, 100, length = 101), SB.mean.(eachcol(clubness_values)))
scatter(range(0, 100, length = 101), SB.mean.(eachcol(clubness_values)))

@profview compute_clubness_values(group_samples_sym[:, :, 1:100])
@code_warntype compute_clubness_values(group_samples_sym[:, :, 1:10])

# θ_rands = [
#     compute_θ(Graphs.adjacency_matrix(Graphs.SimpleGraph(nv, ne)))
#     for _ in 1:n_random_reps
# ]
# mean_θ_rand = SB.mean(θ_rands)

mean_θ_rand = SB.mean(
    compute_θ(Graphs.adjacency_matrix(Graphs.SimpleGraph(nv, ne)))
    for _ in 1:n_random_reps
)

θ_obs / mean_θ_rand

degree_cutoff = .8


Graphs.adjacency_matrix(Graphs.SimpleGraph(nv, ne))

one_sample = group_samples_sym[:, :, 1]
one_degree = degree_samples[:, 1]

richness_cutoff = SB.percentile(one_degree, 80)
rich_nodes_idx = findall(>=(richness_cutoff), one_degree)

n_rich_nodes = length(rich_nodes_idx)
subnet = one_sample[rich_nodes_idx, rich_nodes_idx]
possible_edges = MultilevelGGMSampler.p_to_ne(n_rich_nodes)
θ = (sum(subnet) ÷ 2) / possible_edges


θ_rand

isrich

clubness


#endregion

#region diverse club

import PyCall
# PyCall.@pyinclude("src_python/richclub.py")

function to_igraph(adj)
    igraph = PyCall.pyimport("igraph")
    g_ig = igraph.Graph.Adjacency(adj)
    g_ig.es["weight"] = 1#ones(Int, sum(adj) ÷ 2)
    PyCall.py"""
    $g_ig.vs["name"] = map(str,range($g_ig.vcount()))
    """
    return g_ig
end

group_info_obj = JLD2.jldopen(joinpath(root, "group_object_samples_k_724_p_116_with_prior_probs_cholesky.jld2"))
group_samples = group_info_obj["group_samples"]

group_samples_sym = BitArray(undef, 116, 116, size(group_samples, 2))
for i in axes(group_samples, 2)
    group_samples_sym[:, :, i] = tril_vec_to_sym(group_samples[:, i], -1)
end

function do_walktrap(adj, igraph = PyCall.pyimport("igraph"))

    g_ig = to_igraph(adj)
    walktrap_result = igraph.Graph.community_walktrap(g_ig)
    clusters = walktrap_result.as_clustering(8)
    membership = clusters.membership

    return membership, walktrap_result.optimal_count

end

function compute_participation_coefficients(adj)

    participation_coefficients = ones(size(adj, 1))
    walktrap_result, no_communities = do_walktrap(adj)

    # this step is not very efficient
    index_map = [
        findall(==(i), walktrap_result)
        for i in 0:no_communities - 1
    ]

    for i in eachindex(participation_coefficients)

        value = 0.0
        Kᵢ = sum(view(adj, :, i))
        for s in range(stop = no_communities)
            Kᵢₛ = sum(view(adj, index_map[s], i))
            value += abs2(Kᵢₛ / Kᵢ)
        end

        participation_coefficients[i] -= value
    end

    return participation_coefficients

end

function compute_θ_participation(adj, participation_coefficients, participation_percentile)
    # participation_coefficients = compute_participation_coefficients(adj)
    participation_cutoff = SB.percentile(participation_coefficients, participation_percentile)
    participation_nodes_idx = findall(>=(participation_cutoff), participation_coefficients)
    n_participation_nodes = length(participation_nodes_idx)
    subnet = @view adj[participation_nodes_idx, participation_nodes_idx]
    possible_edges = n_participation_nodes * (n_participation_nodes - 1) ÷ 2
    return (sum(subnet) ÷ 2) / possible_edges
end

# TODO: this function can be generalized to use the same for degree and participation
function compute_clubness_values(adj_group_sym; diverseness_percentile_values = range(0, 100, length = 101))

    # group_samples_sym0 = group_samples_sym
    # group_samples  = view(group_samples_sym0, :, :, 1:10)
    # adj_group_sym = view(group_samples_sym, :, :, 1:10)
    # diverseness_percentile_values = range(0, 100, length = 11)

    clubness_values = Matrix{Float64}(undef, size(adj_group_sym, 3), length(diverseness_percentile_values))

    n_random_reps = 100
    nv = size(adj_group_sym, 1)

    ne_max = nv * (nv - 1) ÷ 2

    unique_edge_counts = sort!(unique(vec(sum(adj_group_sym, dims = 1:2)) .÷ 2))
    # could use OrderedDict
    edge_count_to_idx = Dict(e=>i for (i, e) in enumerate(unique_edge_counts))

    # define here for non threaded version
    # mean_θ_rand = zeros(length(richness_percentile_values))
    # rand_adj_mat = zeros(Int, nv, nv)

    mean_θ_rand_by_participation = Matrix{Float64}(undef, length(diverseness_percentile_values), length(unique_edge_counts))

    prog = ProgressMeter.Progress(length(unique_edge_counts), showspeed = true, desc="Computing θ_rand: ");
    # Threads.@threads
    # (i, ne) = first(enumerate(unique_edge_counts))
    # Threads.@threads
    for (i, ne) in collect(enumerate(unique_edge_counts))

        mean_θ_rand = zeros(length(diverseness_percentile_values))
        # fill!(mean_θ_rand, 0.0)
        linear_edge_indices = similar(1:ne_max, ne)

        rand_adj_mat = zeros(Int, nv, nv)
        for _ in 1:n_random_reps

            fill!(rand_adj_mat, 0)

            SB.sample!(1:ne_max, linear_edge_indices; replace = false)
            for idx in linear_edge_indices
                ii, jj = MultilevelGGMSampler.linear_index_to_lower_triangle_indices(idx, nv)
                rand_adj_mat[ii, jj] = rand_adj_mat[jj, ii] = 1
            end

            participation_coefficients = compute_participation_coefficients(rand_adj_mat)
            for (j, diverseness_percentile) in enumerate(diverseness_percentile_values)
                mean_θ_rand[j] += compute_θ_participation(rand_adj_mat, participation_coefficients, diverseness_percentile)
            end

        end
        mean_θ_rand_by_participation[:, i] .= mean_θ_rand ./ n_random_reps
        ProgressMeter.next!(prog)
    end

    prog = ProgressMeter.Progress(size(adj_group_sym, 3), showspeed = true, desc="Computing θ     : ");
    # Threads.@threads
    for i in axes(adj_group_sym, 3)
        one_sample = @view adj_group_sym[:, :, i]
        participation_coefficients = compute_participation_coefficients(one_sample)
        for (j, diverseness_percentile) in enumerate(diverseness_percentile_values)
            θ_obs = compute_θ_participation(one_sample, participation_coefficients, diverseness_percentile)
            ne = sum(one_sample) ÷ 2
            k = edge_count_to_idx[ne]
            mean_θ_rand = mean_θ_rand_by_participation[j, k]
            clubness_values[i, j] = θ_obs / mean_θ_rand
        end
        ProgressMeter.next!(prog)
    end
    return clubness_values
end


participation_clubness = compute_clubness_values(group_samples_sym)

clubness_values = participation_clubness
clubness_cri = Matrix{Float64}(undef, 2, size(clubness_values, 2))
@views for i in axes(clubness_cri, 2)
    not_nan_idx = findall(!isnan, clubness_values[:, i])
    if length(not_nan_idx) < 3
        clubness_cri[:, i] = [NaN, NaN]
    else
        clubness_cri[:, i] .= SB.quantile(clubness_values[not_nan_idx, i], [.025, .975])
    end
end

# clubness_cri  = [SB.quantile(col, [.025, .975]) for col in eachcol(clubness_values)]
clubness_mean = SB.mean.(eachcol(clubness_values))
diverseness_percentile_values = range(0, 100, length = 101)

not_nan_idx = intersect(
    findall(!isnan, clubness_values[1, :]),
    findall(!isnan, clubness_values[2, :])
)

clubness_figure = Figure()
ax = Axis(clubness_figure[1, 1], xlabel = "Percentile cutoff for the club", ylabel = "Clubness")
band!(diverseness_percentile_values[not_nan_idx], clubness_cri[1, not_nan_idx], clubness_cri[2, not_nan_idx],
    color = Colors.GrayA(.1, .4))
lines!(diverseness_percentile_values, clubness_mean; color = :black)
clubness_figure
save_figs && save(joinpath(figdir, "diverse_clubness.pdf"), clubness_figure)


@profview compute_clubness_values(view(group_samples_sym, :, :, 1:10))

adj = group_samples_sym[:, :, 1]
do_walktrap(adj)

igraph = O
g_ig = to_igraph(adj)
igraph = PyCall.pyimport("igraph")
walktrap_result = igraph.Graph.community_walktrap(g_ig)
clusters = walktrap_result.as_clustering(8)
membership = clusters.membership

map(1:8) do i
    r = walktrap_result.as_clustering(i)
    r.modularity
end

unique_groups = unique(info_df.subnet_order_names)
unique_groups_idx = indexin(info_df.subnet_order_names, unique_groups)
combs = zeros(Int, 8, 8)
for (i1, i2) in zip(unique_groups_idx, membership .+ 1)
    combs[i1, i2] += 1
end

temp_df = DF.DataFrame(
    names   = info_df.subnet_order_names,
    cluster = membership
)

DF.combine(DF.groupby(temp_df, :names), DF.nrow)

#endregion

#region clustering algorithms
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