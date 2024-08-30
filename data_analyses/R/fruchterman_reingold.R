args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1)
  stop("R: argument `results_dir` is missing")

results_dir <- args[1]
if (!dir.exists(results_dir))
  stop(sprintf("R: directory `%s` does not exist", results_dir))

print(sprintf("R: results_dir = %s", results_dir))

adj <- read.csv(file.path(results_dir, "graph_edge_probs.csv"))
obj <- qgraph::qgraph(adj, layout = "spring", repulsion = 3.0, DoNotPlot = TRUE)
write.csv(obj$layout, file = file.path(results_dir, "layout_fr_groupnetwork.csv"), row.names = FALSE)

adj <- read.csv(file.path(results_dir, "graph_mu_vals.csv"))
obj <- qgraph::qgraph(adj, layout = "spring", repulsion = 3.0, DoNotPlot = TRUE)
write.csv(obj$layout, file = file.path(results_dir, "layout_fr_mus_groupnetwork.csv"), row.names = FALSE)


