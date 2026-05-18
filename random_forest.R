library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ranger)

tree_data = readRDS("cleaned_tree_data.rds")
tree_data$sidewalk = as.factor(tree_data$sidewalk)

# ===== DATA SPLITTING =====

set.seed(123)
train_indices = sample(seq_len(nrow(tree_data)), size = floor(0.8 * nrow(tree_data)))
train_data = tree_data[train_indices, ]
remaining_data = tree_data[-train_indices, ]

test_indices = sample(seq_len(nrow(remaining_data)), size = floor(0.5 * nrow(remaining_data)))
test_data = remaining_data[test_indices, ]
dev_data = remaining_data[-test_indices, ]

# ===== SIMPLE RANDOM FOREST =====
# Nothing changed from the raw data 

cat("\n===== SIMPLE RANDOM FOREST =====\n")
cat("Features: 18 original features only\n")
cat("No class weights, no feature engineering\n\n")

rf_model = ranger(
  formula = sidewalk ~ tree_dbh + spc_common + health +
            root_stone + root_grate + root_other +
            trunk_wire + trnk_light + trnk_other +
            brch_light + brch_shoe + brch_other +
            borough + curb_loc + nta_name + steward +
            latitude + longitude,
  data = train_data,
  num.trees = 500,
  probability = TRUE,
  importance = 'none' # change this to permutation for the feature importance graph, otherwise keep as none
)

best_threshold = 0.5 

test_prob = predict(rf_model, data = test_data)$predictions[, "1"]
test_pred = ifelse(test_prob > best_threshold, 1, 0)

tp = sum(test_data$sidewalk == 1 & test_pred == 1)
tn = sum(test_data$sidewalk == 0 & test_pred == 0)
fp = sum(test_data$sidewalk == 0 & test_pred == 1)
fn = sum(test_data$sidewalk == 1 & test_pred == 0)

accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1_score  = 2 * precision * recall / (precision + recall)
f2_score  = 5 * precision * recall / (4 * precision + recall)

cat("===== RESULTS (Threshold:", best_threshold, ") =====\n")
cat("Accuracy: ", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:   ", round(recall, 4), "\n")
cat("F1 Score: ", round(f1_score, 4), "\n")
cat("F2 Score: ", round(f2_score, 4), "\n")

# ===== THRESHOLD OPTIMIZATION ON DEV SET =====

dev_prob = predict(rf_model, data = dev_data)$predictions[, "1"]

best_f1 = 0
best_threshold = 0.5

for (threshold in seq(0.1, 0.9, by = 0.05)) {
  dev_pred = ifelse(dev_prob > threshold, 1, 0)

  tp = sum(dev_data$sidewalk == 1 & dev_pred == 1)
  fp = sum(dev_data$sidewalk == 0 & dev_pred == 1)
  fn = sum(dev_data$sidewalk == 1 & dev_pred == 0)

  precision = ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  recall = ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  f1 = ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)

  if (f1 > best_f1) {
    best_f1 = f1
    best_threshold = threshold
  }
}

cat("Optimal Threshold:", best_threshold, "\n\n")

# ===== EVALUATE ON TEST SET =====

test_prob = predict(rf_model, data = test_data)$predictions[, "1"]
test_pred = ifelse(test_prob > best_threshold, 1, 0)

tp = sum(test_data$sidewalk == 1 & test_pred == 1)
tn = sum(test_data$sidewalk == 0 & test_pred == 0)
fp = sum(test_data$sidewalk == 0 & test_pred == 1)
fn = sum(test_data$sidewalk == 1 & test_pred == 0)

accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1_score  = 2 * precision * recall / (precision + recall)
f2_score  = 5 * precision * recall / (4 * precision + recall)

cat("===== RESULTS (Threshold:", best_threshold, ") =====\n")
cat("Accuracy: ", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:   ", round(recall, 4), "\n")
cat("F1 Score: ", round(f1_score, 4), "\n")
cat("F2 Score: ", round(f2_score, 4), "\n")
