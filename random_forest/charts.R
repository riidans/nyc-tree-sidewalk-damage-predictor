# PERMUTATION IMPORTANCE (REMOVED FEATURES)

library(ggplot2)
library(dplyr)

importance_scores <- rf_model$variable.importance

importance_df <- data.frame(
  Feature = names(importance_scores),
  Importance = as.numeric(importance_scores)
)

importance_df %>%
  arrange(Importance) %>%
  mutate(Feature = factor(Feature, levels = Feature)) %>%
  ggplot(aes(x = Feature, y = Importance, fill = Importance)) +
  geom_col(alpha = 0.9, show.legend = FALSE) + 
  coord_flip() + 
  scale_fill_gradient(low = "#d1d9e6", high = "#2c3e50") + 
  theme_minimal() +
  labs(
    # title = "Random Forest Feature Importance (Removed Features)",
    x = NULL,
    y = "Permutation Importance"
  ) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold")
  )

# ==================

# Install/Load required packages
library(ggplot2)
library(dplyr)
library(ranger)

# 1. Define the 2D boundaries based on the DEV data extent
grid_res <- 150 # Higher number = smoother gradients, but takes slightly longer to render

x_seq <- seq(min(dev_data$longitude, na.rm = TRUE), max(dev_data$longitude, na.rm = TRUE), length.out = grid_res)
y_seq <- seq(min(dev_data$latitude, na.rm = TRUE), max(dev_data$latitude, na.rm = TRUE), length.out = grid_res)

# 2. Create the hypothetical coordinate grid
boundary_grid <- expand.grid(longitude = x_seq, latitude = y_seq)

# 3. Fill in the other 8 features with "Constant" baseline values
boundary_grid <- boundary_grid %>%
  mutate(
    tree_dbh = median(dev_data$tree_dbh, na.rm = TRUE),   # Average sized tree in dev set
    spc_common = "London planetree",                      # Standard species baseline
    health = "Good",                                      # Standard health baseline
    borough = "Queens",                                   
    nta_name = dev_data$nta_name[1],                      # Use a valid NTA from dev set
    curb_loc = "OnCurb",                                  
    steward = "None",
    root_stone = "No"                                     
  )

# Match factor levels EXACTLY to the DEV data so the model doesn't panic
boundary_grid$spc_common <- factor(boundary_grid$spc_common, levels = levels(dev_data$spc_common))
boundary_grid$health <- factor(boundary_grid$health, levels = levels(dev_data$health))
boundary_grid$borough <- factor(boundary_grid$borough, levels = levels(dev_data$borough))
boundary_grid$nta_name <- factor(boundary_grid$nta_name, levels = levels(dev_data$nta_name))
boundary_grid$curb_loc <- factor(boundary_grid$curb_loc, levels = levels(dev_data$curb_loc))
boundary_grid$steward <- factor(boundary_grid$steward, levels = levels(dev_data$steward))
boundary_grid$root_stone <- factor(boundary_grid$root_stone, levels = levels(dev_data$root_stone))

# 4. Predict on the grid (You only need to run the model ONCE!)
grid_predictions <- predict(rf_model, data = boundary_grid, type = "response")
boundary_grid$damage_prob <- grid_predictions$predictions[, "1"]

# 5. Sample the unseen DEV data (Run ONCE so the dots stay perfectly still across all 3 maps)
set.seed(42) 
sample_dev_trees <- dev_data %>% sample_n(min(2000, nrow(dev_data)))

# 6. Build a reusable plotting function
create_threshold_map <- function(threshold_val, metric_name) {
  
  # Create a temporary column that classifies Safe vs Danger based on this specific threshold
  # FIXED: Matched the text "Damaged Boundary" exactly to the scale_fill_manual below
  temp_grid <- boundary_grid %>%
    mutate(risk_zone = ifelse(damage_prob >= threshold_val, "Damaged Boundary", "Not Damaged Boundary"))
  
  # Build and return the plot
  p <- ggplot() +
    # --- BACKGROUND (Solid Colors) ---
    geom_raster(data = temp_grid, aes(x = longitude, y = latitude, fill = risk_zone), alpha = 0.5) +
    
    scale_fill_manual(
      values = c("Not Damaged Boundary" = "#A9CCE3", "Damaged Boundary" = "#F5B7B1"), 
      name = paste("Threshold =", threshold_val)
    ) +
    
    # --- FOREGROUND (Tree Dots) ---
    geom_point(data = sample_dev_trees, aes(x = longitude, y = latitude), 
               color = "black", size = 1.8) +
    geom_point(data = sample_dev_trees, aes(x = longitude, y = latitude, color = sidewalk), 
               size = 1.0) +
    
    scale_color_manual(
      # Ensure "0" and "1" match your actual dev_data factor levels
      values = c("0" = "white", "1" = "black"), 
      name = "2000 Samples from Dev Data",
      labels = c("0" = "Not Damaged", "1" = "Damaged") 
    ) +
    
    # NEW: Add a black outline specifically to the boxes in the 'fill' legend
    guides(
      # order = 1 locks it to the top. override.aes keeps the black borders.
      fill = guide_legend(order = 1, override.aes = list(color = "black", linewidth = 0.5)),
      # order = 2 locks the tree dots to the bottom.
      color = guide_legend(order = 2)
    ) +
    
    # --- CROP & AESTHETICS ---
    coord_cartesian(
      xlim = c(min(dev_data$longitude, na.rm=TRUE) + 0.05, max(dev_data$longitude, na.rm=TRUE)),
      ylim = c(min(dev_data$latitude, na.rm=TRUE), max(dev_data$latitude, na.rm=TRUE) - 0.02)
    ) +
    
    theme_minimal(base_size = 14) +
    labs(
      title = paste("Spatial Decision Boundary:", metric_name),
      x = NULL, y = NULL 
    ) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      panel.grid = element_blank(), 
      plot.background = element_rect(fill = "white", color = NA),
      
      # NEW: Move the legend to the top left (x=0.18, y=0.85 roughly places it over New Jersey)
      legend.position = c(0.18, 0.82),
      # NEW: Give the legend a semi-transparent white background with a border so it stands out from the map
      legend.background = element_rect(fill = alpha("white", 0.85), color = "black", linewidth = 0.3),
      legend.margin = margin(8, 8, 8, 8) # Adds a little breathing room inside the legend box
    )
  
  return(p)
}

# 7. Generate your three separate plots!
plot_default <- create_threshold_map(0.50, "Default (0.50)")
plot_f1      <- create_threshold_map(0.43, "Optimized F1 (0.43)")
plot_f2      <- create_threshold_map(0.22, "Optimized F2 (0.22)")

plot_continuous <- ggplot() +
  # --- BACKGROUND ---
  geom_raster(data = boundary_grid, aes(x = longitude, y = latitude, fill = damage_prob), alpha = 0.85) +
  
  scale_fill_gradientn(
    colors = c("#2980B9", "white", "#E74C3C"), 
    values = scales::rescale(c(0, 0.5, 1)), 
    name = "Raw Probability\nof Damage",
    limits = c(0, 1),
    oob = scales::squish
  ) +
  
  # --- FOREGROUND ---
  geom_point(data = sample_dev_trees, aes(x = longitude, y = latitude), color = "black", size = 1.8) +
  geom_point(data = sample_dev_trees, aes(x = longitude, y = latitude, color = sidewalk), size = 1.0) +
  
  scale_color_manual(
    values = c("0" = "white", "1" = "black"), 
    name = "2000 Samples from Dev Data",
    labels = c("0" = "Not Damaged", "1" = "Damaged")
  ) +
  
  # --- NEW: LOCK THE LEGEND ORDER ---
  # guide_colorbar is used here because damage_prob is a continuous gradient
  guides(
    fill = guide_colorbar(order = 1),
    color = guide_legend(order = 2)
  ) +
  
  # --- CROP & AESTHETICS ---
  coord_cartesian(
    xlim = c(min(dev_data$longitude, na.rm=TRUE) + 0.05, max(dev_data$longitude, na.rm=TRUE)),
    ylim = c(min(dev_data$latitude, na.rm=TRUE), max(dev_data$latitude, na.rm=TRUE) - 0.02)
  ) +
  
  theme_minimal(base_size = 14) +
  labs(
    title = "Spatial Decision Boundary (Continuous)",
    x = NULL, y = NULL 
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    panel.grid = element_blank(), 
    plot.background = element_rect(fill = "white", color = NA),
    
    # --- NEW: MOVE LEGEND INSIDE THE PLOT ---
    legend.position = c(0.18, 0.82),
    legend.background = element_rect(fill = alpha("white", 0.85), color = "black", linewidth = 0.3),
    legend.margin = margin(8, 8, 8, 8) 
  )

# To view them, just run their names in the console:
plot_continuous
plot_default
plot_f1
plot_f2



# ==============================================================

library(pROC)
library(ggplot2)
library(dplyr)
library(xgboost) # Assuming this is what you used for bstSparse

# 1. Logistic Regression Predictions
# (Outputs a simple vector of probabilities)
log_preds <- predict(log_model, newdata = dev_data, type = "response")

# 2. Random Forest Predictions (ranger)
# (Outputs a matrix, we extract the column for class "1")
rf_preds <- predict(rf_model, data = dev_data, type = "response")$predictions[, "1"]

# 3. XGBoost Predictions
# (XGBoost requires the data to be in a sparse matrix format first)
# Note: Replace 'sidewalk' with whatever your target variable is actually named if different
x_dev   <- as.matrix(dev_data[, feature_names])
xgb_preds = predict(bstSparse, x_dev)

# Extract your actual true labels (assuming 0 = Safe, 1 = Damaged)
actual_labels <- dev_data$sidewalk

# Build the ROC mathematical objects
roc_log <- roc(response = actual_labels, predictor = log_preds)
roc_rf  <- roc(response = actual_labels, predictor = rf_preds)
roc_xgb <- roc(response = actual_labels, predictor = xgb_preds)

# Extract the raw AUC scores so we can put them in the legend!
auc_log <- round(auc(roc_log), 3)
auc_rf  <- round(auc(roc_rf), 3)
auc_xgb <- round(auc(roc_xgb), 3)

# Combine into a list for plotting
roc_list <- list(
  "Logistic Regression" = roc_log,
  "Random Forest"       = roc_rf,
  "XGBoost"             = roc_xgb
)

# Plot using ggroc
ggroc(roc_list, linewidth = 1.2) +
  
  # A sleek, modern color palette for the three lines
  scale_color_manual(
    values = c(
      "Logistic Regression" = "#008000", # Green
      "Random Forest"       = "#2980B9", # Blue
      "XGBoost"             = "#E74C3C"  # Red
    ),
    # Embed the calculated AUC scores directly into the legend labels!
    labels = c(
      paste0("Logistic Regression (AUC = ", auc_log, ")"),
      paste0("Random Forest (AUC = ", auc_rf, ")"),
      paste0("XGBoost (AUC = ", auc_xgb, ")")
    ),
    name = "Model Performance"
  ) +
  
  # Add the "Random Guessing" diagonal line (AUC = 0.5)
  geom_abline(intercept = 1, slope = 1, color = "black", linetype = "dashed", alpha = 0.5) +
  
  # Clean aesthetics
  theme_minimal(base_size = 14) +
  labs(
    title = "Model Comparison via ROC",
    x = "True Negative Rate",
    y = "True Positive Rate"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    panel.grid.minor = element_blank(),
    legend.position = c(0.75, 0.25), # Puts the legend inside the bottom-right corner
    legend.background = element_rect(fill = "white", color = "black", linewidth = 0.5),
    legend.text = element_text(size = 12)
  )
