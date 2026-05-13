library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
tree_data <- readRDS("../cleaned_tree_data.rds")

library(xgboost)

set.seed(123)
train_indices <- sample(seq_len(nrow(tree_data)), size = floor(0.8 * nrow(tree_data)))
train_data <- tree_data[train_indices, ]
remaining_data <- tree_data[-train_indices, ]

test_indices <- sample(seq_len(nrow(remaining_data)), size = floor(0.5 * nrow(remaining_data)))
test_data <- remaining_data[test_indices, ]
dev_data <- remaining_data[-test_indices, ]

numeric_cols <- sapply(train_data, function(x) is.numeric(x) || is.logical(x))

feature_names <- names(train_data)[numeric_cols & names(train_data) != "sidewalk"]

x_train <- as.matrix(train_data[, feature_names])
x_dev   <- as.matrix(dev_data[, feature_names])
x_test  <- as.matrix(test_data[, feature_names])

y_train <- as.factor(train_data$sidewalk)
y_dev <- as.factor(dev_data$sidewalk)

bstSparse <- xgboost(
  x = x_train, 
  y = y_train, 
  objective = "binary:logistic",
)

test_prob = predict(bstSparse, x_dev)
test_pred = ifelse(test_prob > 0.5, 1, 0)

tp = sum(dev_data$sidewalk == 1 & test_pred == 1)
tn = sum(dev_data$sidewalk == 0 & test_pred == 0)
fp = sum(dev_data$sidewalk == 0 & test_pred == 1)
fn = sum(dev_data$sidewalk == 1 & test_pred == 0)

accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * (precision * recall) / (precision + recall)
f2 = 5 * (precision * recall) / (4 * precision + recall)

cat("XGBoost (Baseline) Metrics \n", 
    "Accuracy: ", round(accuracy, 4), "\n", 
    "Precision:", round(precision, 4), "\n", 
    "Recall:   ", round(recall, 4), "\n", 
    "F1 Score: ", round(f1, 4), "\n")