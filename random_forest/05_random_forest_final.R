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

rf_model = ranger(
  formula = sidewalk ~ tree_dbh + spc_common + health +
            root_stone + root_grate + root_other +
            trunk_wire + trnk_light + trnk_other +
            brch_light + brch_shoe + brch_other +
            borough + curb_loc + nta_name + steward +
            latitude + longitude,
  data = train_data,
  mtry = 16, # Taken from 04_random_forest_tuning
  min.node.size = 15, # Taken from 04_random_forest_tuning
  num.trees = 500,
  probability = TRUE,
  case.weights = ifelse(train_data$sidewalk == "1", class_weights["1"], class_weights["0"]),
  importance = "none",
  # change this to permutation for the feature importance graph, otherwise keep as none
)

optimal_threshold = 0.45 # Taken from 03_random_forest_weights

# RANDOM FOREST MODEL METRICS ON TEST SET

test_prob = predict(rf_model, data = test_data)$predictions[, "1"]
test_pred = ifelse(test_prob > optimal_threshold, 1, 0)

tp = sum(test_data$sidewalk == 1 & test_pred == 1)
tn = sum(test_data$sidewalk == 0 & test_pred == 0)
fp = sum(test_data$sidewalk == 0 & test_pred == 1)
fn = sum(test_data$sidewalk == 1 & test_pred == 0)

accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1  = 2 * (precision * recall) / (precision + recall)

cat("Random Forest + Optimized Threshold (", optimal_threshold, ") + Class Weights + Tuned Hyperparameters Metrics \n", 
"Accuracy: ", round(accuracy, 4), "\n", 
"Precision:", round(precision, 4), "\n", 
"Recall:   ", round(recall, 4), "\n", 
"F1 Score: ", round(f1, 4), "\n")  