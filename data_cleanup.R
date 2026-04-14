library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))  

raw_tree_data = read.csv("./2015_Street_Tree_Census_-_Tree_Data_20260405.csv") 

# Relevant columns outlined in project proposal
cleaned_tree_data = raw_tree_data[, c("spc_common", "spc_latin", "tree_dbh", "health", "status", 
                  "root_stone", "root_grate", "root_other", 
                  "trunk_wire", "trnk_light", "trnk_other", 
                  "brch_light", "brch_shoe", "brch_other", "curb_loc", 
                  "borough", "nta_name", "latitude", "longitude", "sidewalk", "steward")]

# Cleaning data 
cleaned_tree_data[cleaned_tree_data == ""] = NA
cleaned_tree_data = subset(cleaned_tree_data, status == "Alive" & !is.na(spc_common) & !is.na(health) & !is.na(sidewalk))

cleaned_tree_data$status <- NULL
cleaned_tree_data$spc_common = as.factor(cleaned_tree_data$spc_common)
cleaned_tree_data$health = as.factor(cleaned_tree_data$health)
cleaned_tree_data$steward = as.factor(cleaned_tree_data$steward)
cleaned_tree_data$nta_name = as.factor(cleaned_tree_data$nta_name)

# Encoding target variable to a binary output
cleaned_tree_data$sidewalk = ifelse(cleaned_tree_data$sidewalk == "Damage", 1, 0)

# Binary input features
binary_inputs = c("root_stone", "root_grate", "root_other", 
                  "trunk_wire", "trnk_light", "trnk_other", 
                  "brch_light", "brch_shoe", "brch_other")

for (col in binary_inputs) {
  cleaned_tree_data[[col]] = ifelse(cleaned_tree_data[[col]] == "Yes", 1, 0)
}

# Categorical features
cleaned_tree_data$borough = as.factor(cleaned_tree_data$borough)
cleaned_tree_data$health = as.factor(cleaned_tree_data$health)
cleaned_tree_data$curb_loc = as.factor(cleaned_tree_data$curb_loc)

# Cleaning up 130 species into top 20 most common species
top_20 = names(sort(table(cleaned_tree_data$spc_common), decreasing = TRUE)[1:20])
cleaned_tree_data$spc_common = as.character(cleaned_tree_data$spc_common)
cleaned_tree_data$spc_common = ifelse(cleaned_tree_data$spc_common %in% top_20, 
                                      cleaned_tree_data$spc_common, "Other")
cleaned_tree_data$spc_common[is.na(cleaned_tree_data$spc_common) | cleaned_tree_data$spc_common == ""] = "Other"
cleaned_tree_data$spc_common = as.factor(cleaned_tree_data$spc_common)

# Numerical features 
numeric_features = c('tree_dbh', 'latitude', 'longitude')

for (col in numeric_features) {
  cleaned_tree_data = cleaned_tree_data[!is.na(cleaned_tree_data[[col]]), ]
  cleaned_tree_data[[col]] = as.numeric(cleaned_tree_data[[col]])
}
saveRDS(cleaned_tree_data, file = "cleaned_tree_data.rds")
tree_data = cleaned_tree_data