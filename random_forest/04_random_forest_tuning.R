library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
tree_data = readRDS("../MTH4330/cleaned_tree_data.rds")

library(pROC)

library(ranger)
tree_data$sidewalk = as.factor(tree_data$sidewalk)

# DATA SPLIT
# 80 train / 10 dev / 10 test

set.seed(123)
train_indices = sample(seq_len(nrow(tree_data)), size = floor(0.8 * nrow(tree_data)))
train_data = tree_data[train_indices, ]
remaining_data = tree_data[-train_indices, ]

test_indices = sample(seq_len(nrow(remaining_data)), size = floor(0.5 * nrow(remaining_data)))
test_data = remaining_data[test_indices, ]
dev_data = remaining_data[-test_indices, ]

# CLASS WEIGHTS
# Should target class imbalance

class_dist = table(train_data$sidewalk)
class_weights = c("0" = 1, "1" = as.numeric(class_dist[1] / class_dist[2]))

tuning_grid = expand.grid(
  mtry = c(2, 4, 8, 12, 16),           
  min_node_size = c(1, 5, 10, 15, 20), 
  auc = 0
)

# Changed hyperparameter tuning to measure AUC rather than optimizing for a certain threshold 

for(i in 1:nrow(tuning_grid)) {
  cat("[", format(Sys.time(), "%H:%M:%S"), "] Starting iteration", i, "of", nrow(tuning_grid), "\n")

  temp_model <- ranger( formula = sidewalk ~ tree_dbh + spc_common + health + 
              root_stone + root_grate + root_other + 
              trunk_wire + trnk_light + trnk_other + 
              brch_light + brch_shoe + brch_other + 
              borough + curb_loc + nta_name + steward +
              latitude + longitude, 
      data = train_data,
      num.trees = 500,
      mtry = tuning_grid$mtry[i],
      min.node.size = tuning_grid$min_node_size[i],
      probability = TRUE,
      case.weights = ifelse(train_data$sidewalk == "1", class_weights["1"], class_weights["0"]),
      importance = 'none' 
  )

  dev_prob = predict(temp_model, data = dev_data, type = "response")$predictions[, "1"]
  
  roc_obj = roc(dev_data$sidewalk, dev_prob, quiet = TRUE)
  tuning_grid$auc[i] = auc(roc_obj)

  cat("[", format(Sys.time(), "%H:%M:%S"), "] Finished iteration", i, "of", nrow(tuning_grid), "\n")
  cat("AUC for this iteration:", tuning_grid$auc[i], "\n")
}

tuning_grid = tuning_grid[order(-tuning_grid$auc), ]
saveRDS(tuning_grid, file = "tuning_grid_byauc.rds")