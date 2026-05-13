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
  importance = "none"
  # change this to permutation for the feature importance graph, otherwise keep as none
)

# RANDOM FOREST MODEL METRICS ON DEV SET
# Test set is reserved for final evaluation in 05_random_forest_final.R

dev_prob = predict(rf_model, data = dev_data)$predictions[, "1"]
dev_pred = ifelse(dev_prob > 0.5, 1, 0)

tp = sum(dev_data$sidewalk == 1 & dev_pred == 1)
tn = sum(dev_data$sidewalk == 0 & dev_pred == 0)
fp = sum(dev_data$sidewalk == 0 & dev_pred == 1)
fn = sum(dev_data$sidewalk == 1 & dev_pred == 0)

accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1  = 2 * precision * recall / (precision + recall)
f2 = 5 * (precision * recall) / ((4 * precision) + recall)

cat("Random Forest (Baseline) Dev Metrics \n",
"Accuracy: ", round(accuracy, 4), "\n",
"Precision:", round(precision, 4), "\n",
"Recall:   ", round(recall, 4), "\n",
"F1 Score: ", round(f1, 4), "\n",
"F2 Score: ", round(f2, 4), "\n")


