library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
tree_data = readRDS("../MTH4330/cleaned_tree_data.rds")

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

rf_model = ranger(
  formula = sidewalk ~ tree_dbh + spc_common + health +
            root_stone + root_grate + root_other +
            trunk_wire + trnk_light + trnk_other +
            brch_light + brch_shoe + brch_other +
            borough + curb_loc + nta_name + steward +
            latitude + longitude,
  data = train_data,
  mtry = 12, # Taken from 04_random_forest_tuning
  min.node.size = 15, # Taken from 04_random_forest_tuning
  num.trees = 500,
  probability = TRUE,
  case.weights = ifelse(train_data$sidewalk == "1", class_weights["1"], class_weights["0"]),
  importance = "none",
  # change this to permutation for the feature importance graph, otherwise keep as none
)

dev_prob = predict(rf_model, data = dev_data)$predictions[, "1"]

metrics_table = data.frame()

for (threshold in seq(0.1, 0.9, by = 0.01)) {
  dev_pred = ifelse(dev_prob > threshold, 1, 0)

  tp = sum(dev_data$sidewalk == 1 & dev_pred == 1)
  tn = sum(dev_data$sidewalk == 0 & dev_pred == 0)
  fp = sum(dev_data$sidewalk == 0 & dev_pred == 1)
  fn = sum(dev_data$sidewalk == 1 & dev_pred == 0)

  accuracy  = (tp + tn) / (tp + tn + fp + fn)
  precision = tp / (tp + fp)
  recall    = tp / (tp + fn)
  f1  = 2 * (precision * recall) / (precision + recall)
  f2 = 5 * (precision * recall) / ((4 * precision) + recall)

  current_row = data.frame(threshold, accuracy, precision, recall, f1, f2)
  metrics_table = rbind(metrics_table, current_row)
}

# Takes the threshold with the highest F1 score metric

optimal_threshold_f1 = metrics_table[which.max(metrics_table$f1), ]$threshold #0.46
optimal_threshold_f2 = metrics_table[which.max(metrics_table$f2), ]$threshold #0.23

# RANDOM FOREST MODEL METRICS ON TEST SET

test_prob = predict(rf_model, data = test_data)$predictions[, "1"]
test_pred = ifelse(test_prob > optimal_threshold_f1, 1, 0)

tp = sum(test_data$sidewalk == 1 & test_pred == 1)
tn = sum(test_data$sidewalk == 0 & test_pred == 0)
fp = sum(test_data$sidewalk == 0 & test_pred == 1)
fn = sum(test_data$sidewalk == 1 & test_pred == 0)

accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1  = 2 * (precision * recall) / (precision + recall)
f2 = 5 * (precision * recall) / ((4 * precision) + recall)

cat("Random Forest + Optimized Threshold for F1 (", optimal_threshold_f1, ")\n", 
"Accuracy: ", round(accuracy, 4), "\n", 
"Precision:", round(precision, 4), "\n", 
"Recall:   ", round(recall, 4), "\n", 
"F1 Score: ", round(f1, 4), "\n",
"F2 Score: ", round(f2, 4), "\n")  

# ==============================================================

test_pred = ifelse(test_prob > optimal_threshold_f2, 1, 0)

tp = sum(test_data$sidewalk == 1 & test_pred == 1)
tn = sum(test_data$sidewalk == 0 & test_pred == 0)
fp = sum(test_data$sidewalk == 0 & test_pred == 1)
fn = sum(test_data$sidewalk == 1 & test_pred == 0)

accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1  = 2 * (precision * recall) / (precision + recall)
f2 = 5 * (precision * recall) / ((4 * precision) + recall)
 
cat("Random Forest + Optimized Threshold for F2 (", optimal_threshold_f2, ")\n", 
"Accuracy: ", round(accuracy, 4), "\n", 
"Precision:", round(precision, 4), "\n", 
"Recall:   ", round(recall, 4), "\n", 
"F1 Score: ", round(f1, 4), "\n",
"F2 Score: ", round(f2, 4), "\n")  
