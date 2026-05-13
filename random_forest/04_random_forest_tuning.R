library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
tree_data = readRDS("../cleaned_tree_data.rds")

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
  mtry = c(2, 3, 4, 6, 8),           
  min_node_size = c(5, 10, 15, 20), 
  f2 = 0, 
  threshold = 0,
  auc = 0
)

for(i in 1:nrow(tuning_grid)) {
  cat("[", format(Sys.time(), "%H:%M:%S"), "] Starting iteration", i, "of", nrow(tuning_grid), "\n")

  temp_model <- ranger( formula = sidewalk ~ tree_dbh + spc_common + health + 
              borough + curb_loc + nta_name + steward + 
              latitude + longitude + root_stone, 
      data = train_data,
      num.trees = 500,
      mtry = tuning_grid$mtry[i],
      min.node.size = tuning_grid$min_node_size[i],
      probability = TRUE,
      case.weights = ifelse(train_data$sidewalk == "1", class_weights["1"], class_weights["0"]),
      importance = 'none' 
  )

  dev_prob = predict(temp_model, data = dev_data, type = "response")$predictions[, "1"]
  
  best_local_f2 <- 0
  best_local_threshold <- 0
  
  for (threshold in seq(0.1, 0.9, by = 0.01)) {
    dev_pred <- ifelse(dev_prob > threshold, 1, 0)
    
    tp <- sum(dev_data$sidewalk == 1 & dev_pred == 1)
    fp <- sum(dev_data$sidewalk == 0 & dev_pred == 1)
    fn <- sum(dev_data$sidewalk == 1 & dev_pred == 0)
    
    precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    recall    <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
    
    f2 <- ifelse((4 * precision + recall) == 0, 0, 5 * (precision * recall) / (4 * precision + recall))
    
    if (f2 > best_local_f2) {
      best_local_f2 <- f2
      best_local_threshold <- threshold
    }
  }

  tuning_grid$f2[i] = best_local_f2
  tuning_grid$threshold[i] = best_local_threshold

  roc_obj = roc(dev_data$sidewalk, dev_prob, quiet = TRUE)
  tuning_grid$auc[i] = auc(roc_obj)

  cat("[", format(Sys.time(), "%H:%M:%S"), "] Finished iteration", i, "of", nrow(tuning_grid), "\n")
  cat("F2 for this iteration:", tuning_grid$f2[i], "\n")
  cat("Threshold for this iteration:", tuning_grid$threshold[i], "\n")
}

tuning_grid = tuning_grid[order(-tuning_grid$f2), ]
saveRDS(tuning_grid, file = "tuning_grid_by_f2_NONREPLACE.rds")

tuning_grid = expand.grid(
  mtry = c(2, 3, 4, 6, 8),           
  min_node_size = c(5, 10, 15, 20), 
  sample_replace = c(0.6, 0.7, 0.8),
  f2 = 0, 
  threshold = 0,
  auc = 0
)

for(i in 1:nrow(tuning_grid)) {
  cat("[", format(Sys.time(), "%H:%M:%S"), "] Starting iteration", i, "of", nrow(tuning_grid), "\n")

  temp_model <- ranger( formula = sidewalk ~ tree_dbh + spc_common + health + 
              borough + curb_loc + nta_name + steward + 
              latitude + longitude + root_stone, 
      data = train_data,
      num.trees = 500,
      mtry = tuning_grid$mtry[i],
      min.node.size = tuning_grid$min_node_size[i],
      sample.fraction = tuning_grid$sample_replace[i],
      replace = FALSE, 
      probability = TRUE,
      case.weights = ifelse(train_data$sidewalk == "1", class_weights["1"], class_weights["0"]),
      importance = 'none' 
  )

  dev_prob = predict(temp_model, data = dev_data, type = "response")$predictions[, "1"]
  
  best_local_f2 <- 0
  best_local_threshold <- 0
  
  for (threshold in seq(0.1, 0.9, by = 0.01)) {
    dev_pred <- ifelse(dev_prob > threshold, 1, 0)
    
    tp <- sum(dev_data$sidewalk == 1 & dev_pred == 1)
    fp <- sum(dev_data$sidewalk == 0 & dev_pred == 1)
    fn <- sum(dev_data$sidewalk == 1 & dev_pred == 0)
    
    precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    recall    <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
    
    f2 <- ifelse((4 * precision + recall) == 0, 0, 5 * (precision * recall) / (4 * precision + recall))
    
    if (f2 > best_local_f2) {
      best_local_f2 <- f2
      best_local_threshold <- threshold
    }
  }

  tuning_grid$f2[i] = best_local_f2
  tuning_grid$threshold[i] = best_local_threshold

  roc_obj = roc(dev_data$sidewalk, dev_prob, quiet = TRUE)
  tuning_grid$auc[i] = auc(roc_obj)

  cat("[", format(Sys.time(), "%H:%M:%S"), "] Finished iteration", i, "of", nrow(tuning_grid), "\n")
  cat("F2 for this iteration:", tuning_grid$f2[i], "\n")
  cat("Threshold for this iteration:", tuning_grid$threshold[i], "\n")
}
tuning_grid = tuning_grid[order(-tuning_grid$f2), ]
saveRDS(tuning_grid, file = "tuning_grid_by_f2_REPLACE.rds")