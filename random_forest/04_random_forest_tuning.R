library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
tree_data = readRDS("../cleaned_tree_data.rds")

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

# SKIPPING THRESHOLD OPTIMIZATION PART
# Taking optimal_threshold = 0.45 from 03_random_forest_weights.R

# rf_model = ranger(
#   formula = sidewalk ~ tree_dbh + spc_common + health +
#             root_stone + root_grate + root_other +
#             trunk_wire + trnk_light + trnk_other +
#             brch_light + brch_shoe + brch_other +
#             borough + curb_loc + nta_name + steward +
#             latitude + longitude,
#   data = train_data,
#   num.trees = 500,
#   probability = TRUE,
#   case.weights = ifelse(train_data$sidewalk == "1", class_weights["1"], class_weights["0"]),
#   importance = "none",
#   # change this to permutation for the feature importance graph, otherwise keep as none
# )

# # THRESHOLD OPTIMIZIATION ON DEV SET 
# # Loops through each threshold, and calculates metrics for each 

# dev_prob = predict(rf_model, data = dev_data)$predictions[, "1"]

# metrics_table = data.frame()

# for (threshold in seq(0.1, 0.9, by = 0.05)) {

#   dev_pred = ifelse(dev_prob > threshold, 1, 0)

#   tp = sum(dev_data$sidewalk == 1 & dev_pred == 1)
#   tn = sum(dev_data$sidewalk == 0 & dev_pred == 0)
#   fp = sum(dev_data$sidewalk == 0 & dev_pred == 1)
#   fn = sum(dev_data$sidewalk == 1 & dev_pred == 0)

#   accuracy  = (tp + tn) / (tp + tn + fp + fn)
#   precision = tp / (tp + fp)
#   recall    = tp / (tp + fn)
#   f1  = 2 * (precision * recall) / (precision + recall)

#   current_row = data.frame(threshold, accuracy, precision, recall, f1)
#   metrics_table = rbind(metrics_table, current_row)
# }

# # Takes the threshold with the highest F1 score metric

# optimal_threshold = metrics_table[which.max(metrics_table$f1), ]$threshold

optimal_threshold = 0.45

tuning_grid = expand.grid(
  mtry = c(2, 4, 8, 12, 16),           
  min_node_size = c(1, 5, 10, 15, 20), 
  accuracy = 0,
  precision = 0,
  recall = 0,
  f1 = 0        
)

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
  dev_pred = ifelse(dev_prob > optimal_threshold, 1, 0)

  tp = sum(dev_data$sidewalk == 1 & dev_pred == 1)
  tn = sum(dev_data$sidewalk == 0 & dev_pred == 0)
  fp = sum(dev_data$sidewalk == 0 & dev_pred == 1)
  fn = sum(dev_data$sidewalk == 1 & dev_pred == 0)

  accuracy  = (tp + tn) / (tp + tn + fp + fn)
  precision = tp / (tp + fp)
  recall    = tp / (tp + fn)
  f1  = 2 * (precision * recall) / (precision + recall)

  tuning_grid$accuracy[i] = accuracy
  tuning_grid$precision[i] = precision
  tuning_grid$recall[i] = recall
  tuning_grid$f1[i] = f1

  cat("[", format(Sys.time(), "%H:%M:%S"), "] Finished iteration", i, "of", nrow(tuning_grid), "\n")
}

tuning_grid = tuning_grid[order(-tuning_grid$f1), ]
saveRDS(tuning_grid, file = "tuning_grid.rds")