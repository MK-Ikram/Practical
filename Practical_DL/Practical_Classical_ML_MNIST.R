# Course: Machine Learning for Health Research

#########################################################################
#########################################################################

## Practical III: Classical Machine Learning for MNIST Classification

#########################################################################

### LEARNING OBJECTIVES
# By the end of this practical, you will be able to:
# 1. Apply classical machine learning methods to image classification
# 2. Perform feature extraction and dimensionality reduction
# 3. Compare different ML algorithms (SVM, Random Forest, Logistic Regression)
# 4. Understand the trade-offs between classical ML and deep learning
# 5. Evaluate model performance using various metrics

### PREREQUISITES
# - Basic knowledge of R programming
# - Understanding of machine learning concepts (training/test splits, cross-validation)
# - Familiarity with classification metrics (accuracy, precision, recall, F1-score)

### TIME ESTIMATE
# 60-90 minutes

### IMPORTANT NOTES FOR STUDENTS:
# - This practical complements the deep learning approach
# - Classical ML requires feature engineering (unlike deep learning)
# - Results will be compared with CNN performance from Part I
# - Focus on understanding the feature extraction process

#########################################################################
## Setup: Installing and Loading Required Packages
#########################################################################

# This section installs and loads all necessary packages for classical ML
# e1071: Support Vector Machines and other ML algorithms
# randomForest: Random Forest implementation
# caret: Machine learning utilities and model training
# nnet: Neural networks (for comparison)
# pcaMethods: Principal Component Analysis
# factoextra: PCA visualization
# ggplot2: Advanced plotting
# dplyr: Data manipulation

pkgs <- c("e1071", "randomForest", "caret", "nnet", "pcaMethods", 
          "factoextra", "ggplot2", "dplyr", "reticulate", "tensorflow", "keras")

to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install)) {
  cat("Installing packages:", paste(to_install, collapse = ", "), "\n")
  install.packages(to_install)
}

# Load the required libraries
library(e1071)        # SVM and other ML algorithms
library(randomForest) # Random Forest
library(caret)        # Machine learning utilities
library(nnet)         # Neural networks
library(pcaMethods)   # PCA methods
library(factoextra)   # PCA visualization
library(ggplot2)      # Advanced plotting
library(dplyr)        # Data manipulation
library(reticulate)   # R-Python interface
library(tensorflow)   # For loading MNIST data
library(keras)        # For loading MNIST data

# Python Configuration
use_python(Sys.which("python3"), required = TRUE)

# Reproducibility Setup
set.seed(123)
np <- import("numpy", convert = FALSE)
np$random$seed(123L)
tf$random$set_seed(123L)

#########################################################################
## Loading and Preprocessing MNIST Data
#########################################################################

# Load the MNIST dataset (same as in deep learning practical)
cat("Loading MNIST dataset...\n")
mnist <- dataset_mnist()

# Extract training and test data
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test  <- mnist$test$x
y_test  <- mnist$test$y

# Display dataset information
cat("Dataset information:\n")
cat("Training samples:", nrow(x_train), "\n")
cat("Test samples:", nrow(x_test), "\n")
cat("Image dimensions:", dim(x_train), "\n")

# For classical ML, we need to reshape the data
# Convert 3D arrays (samples, height, width) to 2D matrices (samples, features)
cat("Reshaping data for classical ML...\n")
x_train_flat <- matrix(x_train, nrow = nrow(x_train), ncol = 28*28)
x_test_flat  <- matrix(x_test,  nrow = nrow(x_test),  ncol = 28*28)

# Convert to data frames for easier manipulation
train_df <- data.frame(x_train_flat)
test_df  <- data.frame(x_test_flat)

# Add labels
train_df$label <- y_train
test_df$label  <- y_test

cat("Data shape after flattening:\n")
cat("Training data:", dim(train_df), "\n")
cat("Test data:", dim(test_df), "\n")

#########################################################################
## Feature Engineering and Analysis
#########################################################################

# Classical ML requires feature engineering - let's explore the data
# and create some meaningful features

# 1. Basic statistics of pixel values
cat("Computing basic pixel statistics...\n")
pixel_stats <- data.frame(
  mean_pixel = rowMeans(x_train_flat),
  sd_pixel = apply(x_train_flat, 1, sd),
  min_pixel = apply(x_train_flat, 1, min),
  max_pixel = apply(x_train_flat, 1, max)
)

# Add these features to our dataset
train_df$mean_pixel <- pixel_stats$mean_pixel
train_df$sd_pixel <- pixel_stats$sd_pixel
train_df$min_pixel <- pixel_stats$min_pixel
train_df$max_pixel <- pixel_stats$max_pixel

# 2. Edge density features (simplified)
# Count non-zero pixels (proxy for edge density)
cat("Computing edge density features...\n")
edge_density <- apply(x_train_flat, 1, function(x) sum(x > 0))
train_df$edge_density <- edge_density

# 3. Center vs border features
# Divide image into regions and compute statistics
cat("Computing regional features...\n")
center_pixels <- x_train_flat[, c(98:182, 126:210)]  # Center 7x7 region
border_pixels <- x_train_flat[, -c(98:182, 126:210)] # Border pixels

train_df$center_mean <- rowMeans(center_pixels)
train_df$border_mean <- rowMeans(border_pixels)
train_df$center_border_ratio <- train_df$center_mean / (train_df$border_mean + 1e-8)

# Apply same features to test set
test_pixel_stats <- data.frame(
  mean_pixel = rowMeans(x_test_flat),
  sd_pixel = apply(x_test_flat, 1, sd),
  min_pixel = apply(x_test_flat, 1, min),
  max_pixel = apply(x_test_flat, 1, max)
)

test_df$mean_pixel <- test_pixel_stats$mean_pixel
test_df$sd_pixel <- test_pixel_stats$sd_pixel
test_df$min_pixel <- test_pixel_stats$min_pixel
test_df$max_pixel <- test_pixel_stats$max_pixel

test_edge_density <- apply(x_test_flat, 1, function(x) sum(x > 0))
test_df$edge_density <- test_edge_density

test_center_pixels <- x_test_flat[, c(98:182, 126:210)]
test_border_pixels <- x_test_flat[, -c(98:182, 126:210)]

test_df$center_mean <- rowMeans(test_center_pixels)
test_df$border_mean <- rowMeans(test_border_pixels)
test_df$center_border_ratio <- test_df$center_mean / (test_df$border_mean + 1e-8)

cat("Feature engineering completed!\n")
cat("Number of features:", ncol(train_df) - 1, "\n")  # -1 for label

#########################################################################
## Dimensionality Reduction with PCA
#########################################################################

# With 784 pixel features + engineered features, we have high dimensionality
# Let's use PCA to reduce dimensions while preserving important information

cat("Performing Principal Component Analysis...\n")

# Prepare data for PCA (exclude label column)
pca_data <- train_df[, -ncol(train_df)]  # Remove label column

# Perform PCA
pca_result <- prcomp(pca_data, center = TRUE, scale = TRUE)

# Analyze explained variance
explained_var <- pca_result$sdev^2 / sum(pca_result$sdev^2)
cumulative_var <- cumsum(explained_var)

# Plot explained variance
cat("Plotting PCA explained variance...\n")
plot(1:length(explained_var), cumulative_var, type = "l", 
     xlab = "Number of Components", ylab = "Cumulative Explained Variance",
     main = "PCA: Cumulative Explained Variance")
abline(h = 0.95, col = "red", lty = 2)  # 95% variance line
abline(h = 0.90, col = "blue", lty = 2) # 90% variance line

# Find number of components for 95% variance
n_components_95 <- which(cumulative_var >= 0.95)[1]
n_components_90 <- which(cumulative_var >= 0.90)[1]

cat("Number of components for 95% variance:", n_components_95, "\n")
cat("Number of components for 90% variance:", n_components_90, "\n")

# Use 90% variance for dimensionality reduction
n_components <- n_components_90

# Transform training data
train_pca <- predict(pca_result, train_df[, -ncol(train_df)])[, 1:n_components]
train_pca_df <- data.frame(train_pca)
train_pca_df$label <- train_df$label

# Transform test data
test_pca <- predict(pca_result, test_df[, -ncol(test_df)])[, 1:n_components]
test_pca_df <- data.frame(test_pca)
test_pca_df$label <- test_df$label

cat("Data shape after PCA reduction:\n")
cat("Training data:", dim(train_pca_df), "\n")
cat("Test data:", dim(test_pca_df), "\n")

#########################################################################
## Model Training: Multiple Algorithms
#########################################################################

# Now let's train several classical ML models and compare their performance

cat("Training multiple machine learning models...\n")

# 1. Support Vector Machine (SVM)
cat("Training SVM...\n")
svm_model <- svm(label ~ ., data = train_pca_df, kernel = "radial", 
                 probability = TRUE, scale = FALSE)  # Data already scaled

# 2. Random Forest
cat("Training Random Forest...\n")
rf_model <- randomForest(label ~ ., data = train_pca_df, ntree = 100, 
                        importance = TRUE)

# 3. Logistic Regression (multinomial)
cat("Training Logistic Regression...\n")
# Convert label to factor for multinomial regression
train_pca_df$label <- as.factor(train_pca_df$label)
test_pca_df$label <- as.factor(test_pca_df$label)

lr_model <- multinom(label ~ ., data = train_pca_df, trace = FALSE)

# 4. Simple Neural Network (for comparison)
cat("Training Simple Neural Network...\n")
# Scale data for neural network
train_scaled <- scale(train_pca_df[, -ncol(train_pca_df)])
test_scaled <- scale(test_pca_df[, -ncol(test_pca_df)])

# Create one-hot encoded labels
train_labels <- model.matrix(~ label - 1, data = train_pca_df)
test_labels <- model.matrix(~ label - 1, data = test_pca_df)

# Train neural network
set.seed(123)
nn_model <- nnet(train_scaled, train_labels, size = 50, 
                 decay = 0.01, maxit = 1000, linout = FALSE)

cat("All models trained successfully!\n")

#########################################################################
## Model Evaluation and Comparison
#########################################################################

# Function to calculate accuracy
calculate_accuracy <- function(predictions, true_labels) {
  mean(predictions == true_labels)
}

# Function to calculate confusion matrix and metrics
calculate_metrics <- function(predictions, true_labels, model_name) {
  # Create confusion matrix
  cm <- confusionMatrix(predictions, true_labels)
  
  # Extract metrics
  accuracy <- cm$overall["Accuracy"]
  precision <- mean(cm$byClass[, "Precision"], na.rm = TRUE)
  recall <- mean(cm$byClass[, "Recall"], na.rm = TRUE)
  f1 <- mean(cm$byClass[, "F1"], na.rm = TRUE)
  
  cat(sprintf("%s Results:\n", model_name))
  cat("Accuracy:", round(accuracy, 4), "\n")
  cat("Precision:", round(precision, 4), "\n")
  cat("Recall:", round(recall, 4), "\n")
  cat("F1-Score:", round(f1, 4), "\n")
  cat("---\n")
  
  return(list(accuracy = accuracy, precision = precision, 
              recall = recall, f1 = f1, confusion_matrix = cm))
}

# Make predictions and evaluate models
cat("Evaluating models on test set...\n")

# 1. SVM predictions
svm_pred <- predict(svm_model, test_pca_df[, -ncol(test_pca_df)])
svm_metrics <- calculate_metrics(svm_pred, test_pca_df$label, "SVM")

# 2. Random Forest predictions
rf_pred <- predict(rf_model, test_pca_df[, -ncol(test_pca_df)])
rf_metrics <- calculate_metrics(rf_pred, test_pca_df$label, "Random Forest")

# 3. Logistic Regression predictions
lr_pred <- predict(lr_model, test_pca_df[, -ncol(test_pca_df)])
lr_metrics <- calculate_metrics(lr_pred, test_pca_df$label, "Logistic Regression")

# 4. Neural Network predictions
nn_pred_probs <- predict(nn_model, test_scaled)
nn_pred <- apply(nn_pred_probs, 1, which.max) - 1  # Convert to 0-9
nn_pred <- factor(nn_pred, levels = 0:9)
nn_metrics <- calculate_metrics(nn_pred, test_pca_df$label, "Neural Network")

#########################################################################
## Results Comparison and Visualization
#########################################################################

# Create comparison table
cat("Model Performance Comparison:\n")
cat("============================\n")

results_df <- data.frame(
  Model = c("SVM", "Random Forest", "Logistic Regression", "Neural Network"),
  Accuracy = c(svm_metrics$accuracy, rf_metrics$accuracy, 
               lr_metrics$accuracy, nn_metrics$accuracy),
  Precision = c(svm_metrics$precision, rf_metrics$precision, 
                lr_metrics$precision, nn_metrics$precision),
  Recall = c(svm_metrics$recall, rf_metrics$recall, 
             lr_metrics$recall, nn_metrics$recall),
  F1_Score = c(svm_metrics$f1, rf_metrics$f1, 
               lr_metrics$f1, nn_metrics$f1)
)

print(results_df)

# Visualize results
cat("Creating performance comparison plot...\n")
library(ggplot2)

# Reshape data for plotting
results_long <- tidyr::gather(results_df, Metric, Value, -Model)

# Create comparison plot
p <- ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Model Performance Comparison",
       x = "Model", y = "Score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p)

#########################################################################
## Feature Importance Analysis
#########################################################################

# Analyze feature importance for Random Forest
cat("Analyzing feature importance...\n")

# Get variable importance from Random Forest
importance_scores <- importance(rf_model)
var_importance <- data.frame(
  Feature = rownames(importance_scores),
  Importance = importance_scores[, "MeanDecreaseAccuracy"]
)

# Sort by importance
var_importance <- var_importance[order(-var_importance$Importance), ]

# Plot top 20 features
cat("Plotting top 20 most important features...\n")
top_features <- head(var_importance, 20)

p2 <- ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 20 Most Important Features (Random Forest)",
       x = "Feature", y = "Importance")

print(p2)

#########################################################################
## Comparison with Deep Learning Results
#########################################################################

# Let's compare our classical ML results with deep learning
# (Students should run the deep learning practical first for comparison)

cat("Comparison with Deep Learning:\n")
cat("==============================\n")
cat("Classical ML Results (Best):\n")
best_classical <- results_df[which.max(results_df$Accuracy), ]
cat("Best Model:", best_classical$Model, "\n")
cat("Best Accuracy:", round(best_classical$Accuracy, 4), "\n")
cat("\n")

cat("Expected Deep Learning Results (from Part I):\n")
cat("CNN Accuracy: ~0.98-0.99\n")
cat("CNN Training Time: ~5-10 minutes\n")
cat("\n")

cat("Key Differences:\n")
cat("- Classical ML requires feature engineering\n")
cat("- Deep learning learns features automatically\n")
cat("- Classical ML is faster to train\n")
cat("- Deep learning typically achieves higher accuracy\n")
cat("- Classical ML is more interpretable\n")

#########################################################################
## Advanced Analysis: Error Analysis
#########################################################################

# Let's analyze where our best model makes mistakes
cat("Performing error analysis...\n")

# Use Random Forest as our best model for error analysis
best_model <- rf_model
best_predictions <- rf_pred

# Find misclassified samples
misclassified <- which(best_predictions != test_pca_df$label)
cat("Number of misclassified samples:", length(misclassified), "\n")
cat("Error rate:", round(length(misclassified) / length(best_predictions), 4), "\n")

# Analyze confusion matrix for best model
cat("Confusion Matrix for Random Forest:\n")
print(rf_metrics$confusion_matrix$table)

# Find most confused digit pairs
confusion_table <- rf_metrics$confusion_matrix$table
diag(confusion_table) <- 0  # Remove diagonal (correct predictions)
most_confused <- which(confusion_table == max(confusion_table), arr.ind = TRUE)
cat("Most confused digit pair:", 
    rownames(confusion_table)[most_confused[1]], "vs",
    colnames(confusion_table)[most_confused[2]], "\n")

#########################################################################
## Summary and Key Takeaways
#########################################################################

cat("Summary and Key Takeaways:\n")
cat("==========================\n")

cat("What we learned:\n")
cat("1. Feature engineering is crucial for classical ML\n")
cat("2. PCA helps reduce dimensionality while preserving information\n")
cat("3. Different algorithms have different strengths\n")
cat("4. Random Forest provides good performance and interpretability\n")
cat("5. Classical ML is faster but may have lower accuracy than deep learning\n")

cat("\nFeature Engineering Techniques Used:\n")
cat("- Pixel statistics (mean, std, min, max)\n")
cat("- Edge density estimation\n")
cat("- Regional analysis (center vs border)\n")
cat("- Dimensionality reduction with PCA\n")

cat("\nModel Comparison:\n")
cat("- SVM: Good for high-dimensional data, slower training\n")
cat("- Random Forest: Good performance, interpretable, handles non-linear patterns\n")
cat("- Logistic Regression: Fast, interpretable, linear relationships\n")
cat("- Neural Network: Non-linear patterns, moderate complexity\n")

#########################################################################
## Further Experiments to Try
#########################################################################

cat("Further Experiments to Try:\n")
cat("==========================\n")

cat("1. Feature Engineering:\n")
cat("   - Try different edge detection algorithms\n")
cat("   - Add texture features (GLCM, Haralick)\n")
cat("   - Create shape-based features\n")
cat("   - Add moment-based features\n")

cat("\n2. Dimensionality Reduction:\n")
cat("   - Try different numbers of PCA components\n")
cat("   - Experiment with t-SNE or UMAP\n")
cat("   - Use feature selection methods\n")

cat("\n3. Model Tuning:\n")
cat("   - Use cross-validation for hyperparameter tuning\n")
cat("   - Try ensemble methods (voting, stacking)\n")
cat("   - Experiment with different kernels for SVM\n")

cat("\n4. Data Augmentation:\n")
cat("   - Add noise to training data\n")
cat("   - Apply small rotations/translations\n")
cat("   - Use SMOTE for balanced classes\n")

#########################################################################
## Troubleshooting Guide
#########################################################################

cat("Troubleshooting Guide:\n")
cat("=====================\n")

cat("1. 'Out of memory' error:\n")
cat("   - Reduce number of training samples\n")
cat("   - Use fewer PCA components\n")
cat("   - Try simpler models\n")

cat("\n2. 'Model training is slow':\n")
cat("   - Reduce number of features\n")
cat("   - Use smaller training set for testing\n")
cat("   - Try faster algorithms (Logistic Regression)\n")

cat("\n3. 'Poor model performance':\n")
cat("   - Check feature scaling\n")
cat("   - Try different feature engineering techniques\n")
cat("   - Increase number of PCA components\n")
cat("   - Use cross-validation for model selection\n")

cat("\n4. 'PCA not working well':\n")
cat("   - Check for missing values\n")
cat("   - Ensure data is properly scaled\n")
cat("   - Try different numbers of components\n")

#########################################################################
## END of PRACTICAL
#########################################################################

cat("Congratulations! You've completed the Classical ML practical!\n")
cat("You've learned how to:\n")
cat("- Apply feature engineering to image data\n")
cat("- Use dimensionality reduction techniques\n")
cat("- Train and compare multiple ML algorithms\n")
cat("- Analyze model performance and errors\n")
cat("- Understand trade-offs between classical ML and deep learning\n")

cat("\nNext steps:\n")
cat("- Compare results with the deep learning practical\n")
cat("- Try the suggested experiments above\n")
cat("- Apply these techniques to other image datasets\n")
cat("- Explore more advanced feature engineering methods\n")
