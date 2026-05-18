# Merge Python-generated HDBSCAN features back into cleaned_tree_data.rds.

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    return(dirname(rstudioapi::getActiveDocumentContext()$path))
  }
  return(getwd())
}

script_dir <- get_script_dir()
repo_root <- normalizePath(file.path(script_dir, ".."))

tree_data <- readRDS(file.path(repo_root, "cleaned_tree_data.rds"))
tree_data$row_id <- seq_len(nrow(tree_data))

features_path <- file.path(repo_root, "clustering", "data", "hdbscan_features.csv")
features <- read.csv(features_path, stringsAsFactors = FALSE)

# The split column comes from the exported split file and lets downstream scripts
# reuse exactly the same rows instead of re-sampling independently.
merged <- merge(tree_data, features, by = "row_id", all.x = TRUE, sort = FALSE)
merged <- merged[order(merged$row_id), ]

if (any(is.na(merged$hdbscan_fit_group))) {
  stop("Some rows did not receive HDBSCAN features. Check BOROUGH_CONFIGS coverage.")
}

factor_cols <- grep("^hdbscan_.*(group|cluster)$", names(merged), value = TRUE)
for (col in factor_cols) {
  merged[[col]] <- as.factor(merged[[col]])
}

merged$split <- factor(merged$split, levels = c("train", "dev", "test"))
merged$sidewalk <- as.factor(merged$sidewalk)

merged <- subset(merged, select = -c(latitude, longitude))

out_path <- file.path(repo_root, "cleaned_tree_data_hdbscan.rds")
saveRDS(merged, file = out_path)
cat("Wrote:", out_path, "\n")
cat("Rows by split:\n")
print(table(merged$split))
cat("Rows by HDBSCAN fit group:\n")
print(table(merged$hdbscan_fit_group))
