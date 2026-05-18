# Export the cleaned data with the same train/dev/test split used by the
# current modeling scripts. This creates a stable row_id so Python-generated
# clustering features can be merged back without depending on row order.

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

source_rds <- file.path(repo_root, "cleaned_tree_data.rds")
out_dir <- file.path(repo_root, "clustering", "data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tree_data <- readRDS(source_rds)
tree_data$row_id <- seq_len(nrow(tree_data))

# Kept identical to the existing random-forest split logic
set.seed(123)
all_indices <- seq_len(nrow(tree_data))
train_indices <- sample(all_indices, size = floor(0.8 * nrow(tree_data)))
remaining_indices <- all_indices[-train_indices]

test_relative_indices <- sample(seq_along(remaining_indices),
                                size = floor(0.5 * length(remaining_indices)))
test_indices <- remaining_indices[test_relative_indices]
dev_indices <- remaining_indices[-test_relative_indices]

tree_data$split <- NA_character_
tree_data$split[train_indices] <- "train"
tree_data$split[dev_indices] <- "dev"
tree_data$split[test_indices] <- "test"

split_indices <- list(
  train_indices = train_indices,
  dev_indices = dev_indices,
  test_indices = test_indices
)

manhattan_train <- cbind(tree_data$longitude[which(tree_data$borough == "Manhattan" & tree_data$split == "train")],
                         tree_data$latitude[which(tree_data$borough == "Manhattan" & tree_data$split == "train")])

bronx_train <- cbind(tree_data$longitude[which(tree_data$borough == "Bronx" & tree_data$split == "train")],
                     tree_data$latitude[which(tree_data$borough == "Bronx" & tree_data$split == "train")])

staten_train <- cbind(tree_data$longitude[which(tree_data$borough == "Staten Island" & tree_data$split == "train")],
                      tree_data$latitude[which(tree_data$borough == "Staten Island" & tree_data$split == "train")])

bklyn_queens_train <- cbind(tree_data$longitude[which((tree_data$borough == "Brooklyn" | tree_data$borough == "Queens") & tree_data$split == "train")],
                            tree_data$latitude[which((tree_data$borough == "Brooklyn" | tree_data$borough == "Queens") & tree_data$split == "train")])

dbopt_dir <- file.path(repo_root, "clustering", "dbopt_final", "train_data")
saveRDS(manhattan_train, file = file.path(dbopt_dir, "manhattan_geo_data.rds"))
saveRDS(bronx_train, file = file.path(dbopt_dir, "bronx_geo_data.rds"))
saveRDS(staten_train, file = file.path(dbopt_dir, "staten_geo_data.rds"))
saveRDS(bklyn_queens_train, file = file.path(dbopt_dir, "bklyn_queens_geo_data.rds"))

write.csv(tree_data,
          file = file.path(out_dir, "tree_data_with_split.csv"),
          row.names = FALSE,
          na = "")
saveRDS(split_indices, file = file.path(out_dir, "split_indices.rds"))

cat("Wrote:", file.path(out_dir, "tree_data_with_split.csv"), "\n")
cat("Rows by split:\n")
print(table(tree_data$split))
