library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
tree_data <- readRDS("../cleaned_tree_data.rds")

library(ranger)
tree_data$sidewalk <- as.factor(tree_data$sidewalk)


# steward_num and health_num 
tree_data$steward_num <- ifelse(tree_data$steward == "None",    0,
                                ifelse(tree_data$steward == "1or2",    1,
                                       ifelse(tree_data$steward == "3or4",    2,
                                              ifelse(tree_data$steward == "4orMore", 3, 0))))

tree_data$health_num <- ifelse(tree_data$health == "Poor", 1,
                               ifelse(tree_data$health == "Fair", 2,
                                      ifelse(tree_data$health == "Good", 3, 2)))

# Interaction features
tree_data$care_index      <- tree_data$steward_num * tree_data$health_num
tree_data$neglect_risk    <- tree_data$tree_dbh * (4 - tree_data$steward_num)
tree_data$poor_health_big <- tree_data$tree_dbh * (4 - tree_data$health_num)
tree_data$risk_score      <- tree_data$tree_dbh * (4 - tree_data$steward_num) * (4 - tree_data$health_num)

# DATA SPLIT
set.seed(123)
train_indices <- sample(seq_len(nrow(tree_data)), size = floor(0.8 * nrow(tree_data)))
train_data <- tree_data[train_indices, ]
remaining_data <- tree_data[-train_indices, ]

test_indices <- sample(seq_len(nrow(remaining_data)), size = floor(0.5 * nrow(remaining_data)))
test_data <- remaining_data[test_indices, ]
dev_data <- remaining_data[-test_indices, ]

# CLASS WEIGHTS
class_dist <- table(train_data$sidewalk)
class_weights <- c("0" = 1, "1" = as.numeric(class_dist[1] / class_dist[2]))

# Baseline formula (leakage‑reduced)
base_formula <- sidewalk ~ tree_dbh + spc_common + health + root_stone +
  borough + curb_loc + nta_name + steward + latitude + longitude

# Feature sets to test
feature_sets <- list(
  Baseline            = base_formula,
  "+ care_index"      = update(base_formula, ~ . + care_index),
  "+ neglect_risk"    = update(base_formula, ~ . + neglect_risk),
  "+ poor_health_big" = update(base_formula, ~ . + poor_health_big),
  "+ risk_score"      = update(base_formula, ~ . + risk_score)
)

# Helper to compute metrics at a given threshold
calc_metrics <- function(true_labels, prob, threshold) {
  pred <- ifelse(prob > threshold, 1, 0)
  tp <- sum(true_labels == 1 & pred == 1)
  tn <- sum(true_labels == 0 & pred == 0)
  fp <- sum(true_labels == 0 & pred == 1)
  fn <- sum(true_labels == 1 & pred == 0)
  
  accuracy  <- (tp + tn) / (tp + tn + fp + fn)
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  recall    <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  f1        <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  f2        <- ifelse(((4 * precision) + recall) > 0, 5 * precision * recall / ((4 * precision) + recall), 0)
  
  return(c(F1 = f1, F2 = f2, Precision = precision, Recall = recall))
}

# Optimal thresholds (from the baseline dev‑set optimisation)
threshold_f1 <- 0.46
threshold_f2 <- 0.23

# Train and evaluate
results <- data.frame()

for (nm in names(feature_sets)) {
  cat("Training:", nm, "\n")
  
  set.seed(456)
  rf <- ranger(
    formula = feature_sets[[nm]],
    data = train_data,
    mtry = 6,                
    min.node.size = 5,      
    num.trees = 500,
    probability = TRUE,
    case.weights = ifelse(train_data$sidewalk == "1", class_weights["1"], class_weights["0"]),
    importance = "none"
  )
  
  # Predict on test set
  test_prob <- predict(rf, data = test_data)$predictions[, "1"]
  true_lab  <- as.numeric(as.character(test_data$sidewalk))
  
  # Metrics at both thresholds
  m1 <- calc_metrics(true_lab, test_prob, threshold_f1)
  m2 <- calc_metrics(true_lab, test_prob, threshold_f2)
  
  results <- rbind(results, data.frame(
    Model = nm,
    F1_0.46 = round(m1["F1"], 4),
    F2_0.46 = round(m1["F2"], 4),
    F1_0.23 = round(m2["F1"], 4),
    F2_0.23 = round(m2["F2"], 4),
    stringsAsFactors = FALSE
  ))
}

# Deltas from Baseline
baseline_f1_46 <- results$F1_0.46[results$Model == "Baseline"]
baseline_f2_46 <- results$F2_0.46[results$Model == "Baseline"]
baseline_f1_23 <- results$F1_0.23[results$Model == "Baseline"]
baseline_f2_23 <- results$F2_0.23[results$Model == "Baseline"]

results$dF1_46 <- round(results$F1_0.46 - baseline_f1_46, 4)
results$dF2_46 <- round(results$F2_0.46 - baseline_f2_46, 4)
results$dF1_23 <- round(results$F1_0.23 - baseline_f1_23, 4)
results$dF2_23 <- round(results$F2_0.23 - baseline_f2_23, 4)

# Output
cat("\n========================================\n")
cat("Care Interaction Indices – F1 threshold (0.46)\n")
print(results[, c("Model", "F1_0.46", "F2_0.46", "dF1_46", "dF2_46")], row.names = FALSE)

cat("\nCare Interaction Indices – F2 threshold (0.23)\n")
print(results[, c("Model", "F1_0.23", "F2_0.23", "dF1_23", "dF2_23")], row.names = FALSE)

# write.csv(results, "results_care_interactions.csv", row.names = FALSE)