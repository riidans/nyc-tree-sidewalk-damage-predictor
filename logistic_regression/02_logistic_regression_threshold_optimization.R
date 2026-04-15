library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
tree_data = readRDS("cleaned_tree_data.rds") 

# DATA SPLIT
# 80 train / 10 dev / 10 test

set.seed(123)
train_indices = sample(seq_len(nrow(tree_data)), size = floor(0.8 * nrow(tree_data)))
train_data = tree_data[train_indices, ]
remaining_data = tree_data[-train_indices, ]

test_indices = sample(seq_len(nrow(remaining_data)), size = floor(0.5 * nrow(remaining_data)))
test_data = remaining_data[test_indices, ]
dev_data = remaining_data[-test_indices, ]

log_model = glm(sidewalk ~ tree_dbh + spc_common + health + 
                  root_stone + root_grate + root_other + 
                  trunk_wire + trnk_light + trnk_other + 
                  brch_light + brch_shoe + brch_other + 
                  borough + curb_loc + nta_name + steward + 
                  latitude + longitude,
                  data = train_data, 
                  family = "binomial",
)      

# THRESHOLD OPTIMIZATION 
# Loops through each threshold, and calculates metrics for each (on dev split)

dev_prob = predict(log_model, newdata = dev_data, type = "response")

metrics_table = data.frame()

for (threshold in seq(0.1, 0.9, by = 0.05)){

  dev_pred = ifelse(dev_prob > threshold, 1, 0)

  tp = sum(dev_data$sidewalk == 1 & dev_pred == 1)
  tn = sum(dev_data$sidewalk == 0 & dev_pred == 0)
  fp = sum(dev_data$sidewalk == 0 & dev_pred == 1)
  fn = sum(dev_data$sidewalk == 1 & dev_pred == 0)

  accuracy  = (tp + tn) / (tp + tn + fp + fn)
  precision = tp / (tp + fp)
  recall    = tp / (tp + fn)
  f1  = 2 * (precision * recall) / (precision + recall)

  current_row = data.frame(threshold, accuracy, precision, recall, f1)
  metrics_table = rbind(metrics_table, current_row)
}

# Take the threshold with the highest F1 score metric

optimal_threshold = metrics_table[which.max(metrics_table$f1), ]$threshold

# TEST SPLIT RESULTS

test_prob = predict(log_model, newdata = test_data, type = "response")
test_pred = ifelse(test_prob > optimal_threshold, 1, 0)

tp = sum(test_data$sidewalk == 1 & test_pred == 1)
tn = sum(test_data$sidewalk == 0 & test_pred == 0)
fp = sum(test_data$sidewalk == 0 & test_pred == 1)
fn = sum(test_data$sidewalk == 1 & test_pred == 0)

accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1  = 2 * (precision * recall) / (precision + recall)

cat("Logistic Regression + Optimized Threshold (", optimal_threshold, ") Metrics \n", 
"Accuracy: ", round(accuracy, 4), "\n", 
"Precision:", round(precision, 4), "\n", 
"Recall:   ", round(recall, 4), "\n", 
"F1 Score: ", round(f1, 4), "\n")  
 