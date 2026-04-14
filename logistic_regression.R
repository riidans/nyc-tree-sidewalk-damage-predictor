library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
tree_data = readRDS("cleaned_tree_data.rds") # This is already cleaned, check data_cleanup.R

# ===== LOGISTIC REGRESSION BASELINE =====

set.seed(123)
# 80 train, 10 dev, 10 test
train_indices = sample(seq_len(nrow(tree_data)), size = floor(0.8 * nrow(tree_data)))
train_data = tree_data[train_indices, ]
remaining_data = tree_data[-train_indices, ]

test_indices = sample(seq_len(nrow(remaining_data)), size = floor(0.5 * nrow(remaining_data)))
test_data = remaining_data[test_indices, ]
dev_data = remaining_data[-test_indices, ]

# No major feature engineering or anything, this is just everything thrown in
log_model = glm(sidewalk ~ tree_dbh + spc_common + health + 
                  root_stone + root_grate + root_other + 
                  trunk_wire + trnk_light + trnk_other + 
                  brch_light + brch_shoe + brch_other + 
                  borough + curb_loc + nta_name + steward, 
                  data = train_data, 
                  family = "binomial")

test_prob = predict(log_model, newdata = test_data, type = "response")
test_data$predicted = factor(ifelse(test_prob > 0.5, 1, 0), levels = c(0, 1))

conf_matrix = table(Actual = test_data$sidewalk, Predicted = test_data$predicted)
tp = conf_matrix[2, 2] # Actual 1, Predicted 1
tn = conf_matrix[1, 1] # Actual 0, Predicted 0
fp = conf_matrix[1, 2] # Actual 0, Predicted 1
fn = conf_matrix[2, 1] # Actual 1, Predicted 0

accuracy  = (tp + tn) / sum(conf_matrix)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1_score  = 2 * (precision * recall) / (precision + recall)
f2_score  = 5 * (precision * recall) / ((4 * precision) + recall)

cat(" Accuracy: ", round(accuracy, 4), "\n", "Precision:", round(precision, 4), "\n", "Recall:   ", round(recall, 4), "\n", "F1 Score: ", round(f1_score, 4), "\n", "F2 Score: ", round(f2_score, 4))    

# ===== THRESHOLD OPTIMIZATION ======

dev_prob = predict(log_model, newdata = dev_data, type = "response")

metrics_table = data.frame()

for (threshold in seq(0.1, 0.9, by = 0.05)){

  dev_data$predicted = factor(ifelse(dev_prob > threshold, 1, 0), levels = c(0, 1))

  conf_matrix = table(Actual = dev_data$sidewalk, Predicted = dev_data$predicted)
  tp = conf_matrix[2, 2] # Actual 1, Predicted 1
  tn = conf_matrix[1, 1] # Actual 0, Predicted 0
  fp = conf_matrix[1, 2] # Actual 0, Predicted 1
  fn = conf_matrix[2, 1] # Actual 1, Predicted 0

  accuracy  = (tp + tn) / sum(conf_matrix)
  precision = tp / (tp + fp)
  recall    = tp / (tp + fn)
  f1_score  = 2 * (precision * recall) / (precision + recall)
  f2_score  = 5 * (precision * recall) / ((4 * precision) + recall)

  current_row = data.frame(threshold, accuracy, precision, recall, f1_score, f2_score)
  metrics_table = rbind(metrics_table, current_row)
}

# ===== THRESHOLD OPTIMIZATION BASED ON F1 SCORE =====

optimal_threshold = metrics_table[which.max(metrics_table$f1_score), ]$threshold

test_prob = predict(log_model, newdata = test_data, type = "response")
test_data$predicted = factor(ifelse(test_prob > optimal_threshold, 1, 0), levels = c(0, 1))

conf_matrix = table(Actual = test_data$sidewalk, Predicted = test_data$predicted)
tp = conf_matrix[2, 2] # Actual 1, Predicted 1
tn = conf_matrix[1, 1] # Actual 0, Predicted 0
fp = conf_matrix[1, 2] # Actual 0, Predicted 1
fn = conf_matrix[2, 1] # Actual 1, Predicted 0

accuracy  = (tp + tn) / sum(conf_matrix)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1_score  = 2 * (precision * recall) / (precision + recall)
f2_score  = 5 * (precision * recall) / ((4 * precision) + recall)

cat(" Accuracy: ", round(accuracy, 4), "\n", "Precision:", round(precision, 4), "\n", "Recall:   ", round(recall, 4), "\n", "F1 Score: ", round(f1_score, 4), "\n", "F2 Score: ", round(f2_score, 4))    

# ===== THRESHOLD OPTIMIZATION BASED ON F2 SCORE =====

optimal_threshold = metrics_table[which.max(metrics_table$f2_score), ]$threshold

test_prob = predict(log_model, newdata = test_data, type = "response")
test_data$predicted = factor(ifelse(test_prob > optimal_threshold, 1, 0), levels = c(0, 1))

conf_matrix = table(Actual = test_data$sidewalk, Predicted = test_data$predicted)
tp = conf_matrix[2, 2] # Actual 1, Predicted 1
tn = conf_matrix[1, 1] # Actual 0, Predicted 0
fp = conf_matrix[1, 2] # Actual 0, Predicted 1
fn = conf_matrix[2, 1] # Actual 1, Predicted 0

accuracy  = (tp + tn) / sum(conf_matrix)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1_score  = 2 * (precision * recall) / (precision + recall)
f2_score  = 5 * (precision * recall) / ((4 * precision) + recall)

cat(" Accuracy: ", round(accuracy, 4), "\n", "Precision:", round(precision, 4), "\n", "Recall:   ", round(recall, 4), "\n", "F1 Score: ", round(f1_score, 4), "\n", "F2 Score: ", round(f2_score, 4))    


