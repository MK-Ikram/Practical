

# Course: Machine Learning for Health Research

#########################################################################
#########################################################################

## Practical III: Neural Networks 

#########################################################################

### LEARNING OBJECTIVES
# By the end of this practical, you will be able to:
# 1. Set up TensorFlow/Keras in R for deep learning
# 2. Preprocess image data for neural networks
# 3. Build and train a Convolutional Neural Network (CNN)
# 4. Evaluate model performance using accuracy, loss plots, and confusion matrices
# 5. Apply deep learning to real-world medical imaging data

### PREREQUISITES
# - Basic knowledge of R programming
# - Understanding of machine learning concepts (training/test splits, overfitting)
# - Familiarity with image data and preprocessing
# - Python 3.x installed on your system (TensorFlow dependency)


### This practical contains two parts:
# Part I: Neural network on MNIST dataset (handwritten digits)
# Part II: Neural network for health-related dataset using MRI images (brain tumors)

### IMPORTANT NOTES FOR STUDENTS:
# - Run each code block sequentially - some blocks depend on previous ones
# - Pay attention to the comments explaining what each step does
# - If you encounter errors, check the troubleshooting section at the end
# - The MRI dataset is large - ensure you have sufficient memory and storage


#########################################################################
## Setup: Installing and Loading Required Packages
#########################################################################

# This section installs and loads all necessary packages for deep learning
# reticulate: Interface between R and Python
# tensorflow: Deep learning framework
# keras: High-level neural network API
# magrittr: Provides the %>% pipe operator for cleaner code
# abind: For combining arrays (used in Part II)
# jpeg: For reading JPEG image files

pkgs <- c("reticulate","tensorflow","keras","magrittr","abind","jpeg")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install)) {
  cat("Installing packages:", paste(to_install, collapse = ", "), "\n")
  install.packages(to_install)
}

# Load the required libraries
library(reticulate)    # R-Python interface
library(tensorflow)    # Deep learning framework
library(keras)         # High-level neural network API
library(magrittr)      # Provides %>% operator for cleaner code

# Python Configuration
# TensorFlow requires Python. This line tells R which Python installation to use
# If you get an error here, make sure Python 3.x is installed on your system
use_python(Sys.which("python3"), required = TRUE)

# macOS Apple Silicon users (optional): 
# If you're on a Mac with Apple Silicon (M1/M2), uncomment these lines for better performance:
# reticulate::install_miniconda()
# conda_create("r-tf")
# use_condaenv("r-tf", required=TRUE)
# py_install(c("tensorflow-macos","tensorflow-metal","h5py","numpy<2"), pip=TRUE)

# Reproducibility Setup
# Setting random seeds ensures that your results are reproducible
# This is crucial for scientific research and debugging
set.seed(123)                                    # R random seed
np <- import("numpy", convert = FALSE)           # Import NumPy
np$random$seed(123L)                             # NumPy random seed
tf$random$set_seed(123L)                         # TensorFlow random seed

# System Information
# This helps with troubleshooting if you encounter platform-specific issues
cat("R architecture:", R.version$arch, "\n")
cat("System machine:", Sys.info()[["machine"]], "\n")

#########################################################################
## TensorFlow / Keras Verification Test
#########################################################################

# This section verifies that TensorFlow is working correctly
# If these commands run without errors, your setup is ready!

cat("TensorFlow version:", tf$version$VERSION, "\n")

# Simple test: add two numbers using TensorFlow
# This should return 3. If it works, TensorFlow is functioning properly
test_result <- tf$math$add(1L, 2L)$numpy()
cat("TensorFlow test (1 + 2 =", test_result, "):", ifelse(test_result == 3, "SUCCESS", "FAILED"), "\n")  

#########################################################################
## Part I: MNIST Dataset - Handwritten Digit Recognition
#########################################################################

# MNIST is a classic dataset containing 70,000 handwritten digits (0-9)
# It's perfect for learning deep learning because:
# - It's well-structured and clean
# - Results are easy to interpret
# - Training is relatively fast
# - It demonstrates key concepts without overwhelming complexity

# Load the MNIST dataset
# This downloads the data automatically if not already present
mnist <- dataset_mnist()

# Explore the structure of the dataset
# You'll see it contains train and test sets, each with images (x) and labels (y)
cat("MNIST dataset structure:\n")
str(mnist)



# Extract training and test data
# x_train/x_test: Images (features) - each image is 28x28 pixels
# y_train/y_test: Labels (targets) - digit values 0-9

x_train <- mnist$train$x  # Training images
y_train <- mnist$train$y  # Training labels
x_test  <- mnist$test$x   # Test images
y_test  <- mnist$test$y   # Test labels

# Explore the dataset dimensions
cat("Dataset sizes:\n")
cat("Training samples:", nrow(x_train), "\n")  # Should be 60,000
cat("Test samples:", nrow(x_test), "\n")       # Should be 10,000

# Check the dimensions of individual images
# Each image is 28x28 pixels (grayscale)
cat("Image dimensions:", dim(x_train), "\n")   # Should be 60000 28 28



# Function to visualize sample images
# This helps us understand what the data looks like before training
plot_samples <- function(x, y) {
  par(mfrow = c(1, 5), mar = c(1, 1, 2, 1))  # layout: 1 row, 5 columns
  for (i in 1:5) {
    # Display each image with its corresponding label
    image(t(apply(x[i,,], 2, rev)), col = gray.colors(256), axes = FALSE, 
          main = paste("Label:", y[i]))
  }
}

# Visualize 5 sample images from the training set
# This shows us what handwritten digits look like in the dataset
cat("Displaying 5 sample images from the training set:\n")
plot_samples(x_train, y_train)


#########################################################################
## Data Preprocessing for Neural Networks
#########################################################################

# Before training a neural network, we need to prepare the data properly
# This involves normalization and reshaping

# Step 1: Normalize pixel values
# Original pixel values range from 0-255 (8-bit grayscale)
# Neural networks work better with values between 0-1
x_train <- x_train / 255  # Divide by 255 to get values in [0,1]
x_test  <- x_test / 255

# Step 2: Reshape data for CNN
# CNNs expect 4D input: (samples, height, width, channels)
# Original shape: (samples, 28, 28)
# New shape: (samples, 28, 28, 1) - where 1 is the number of color channels (grayscale)
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test  <- array_reshape(x_test,  c(nrow(x_test),  28, 28, 1))

# Verify the new shapes
cat("Data shapes after preprocessing:\n")
cat("x_train shape:", dim(x_train), "\n")  # Should be 60000 28 28 1
cat("x_test shape:",  dim(x_test),  "\n")  # Should be 10000 28 28 1


# Step 3: Convert labels to one-hot encoding
# Neural networks for classification need labels in one-hot format
# Instead of single numbers (0,1,2,...,9), we create binary vectors
# Example: digit 3 becomes [0,0,0,1,0,0,0,0,0,0]

y_train <- tf$keras$utils$to_categorical(y_train, num_classes = 10L)
y_test  <- tf$keras$utils$to_categorical(y_test,  num_classes = 10L)

# Verify the label shapes
cat("Label shapes after one-hot encoding:\n")
cat("y_train shape:", dim(y_train), "\n")  # Should be 60000 10
cat("y_test shape:",  dim(y_test),  "\n")  # Should be 10000 10




#########################################################################
## Building the Convolutional Neural Network (CNN)
#########################################################################

# Now we'll build a CNN architecture suitable for image classification
# CNNs are particularly effective for image data because they can learn:
# - Local patterns (edges, textures)
# - Hierarchical features (combinations of simple patterns)
# - Translation-invariant features (patterns that appear anywhere in the image)

# Define the input layer
# Shape: (28, 28, 1) = (height, width, channels)
inputs <- layer_input(shape = c(28, 28, 1))

# Build the CNN architecture layer by layer
outputs <- inputs %>%
  # First convolutional layer: learns 32 different feature maps
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  # Max pooling: reduces spatial dimensions by half, keeps important features
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # Second convolutional layer: learns 64 more complex feature maps
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  # Another max pooling layer
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # Flatten: convert 2D feature maps to 1D vector for dense layers
  layer_flatten() %>%
  # Dense layer: fully connected layer with 128 neurons
  layer_dense(units = 128, activation = "relu") %>%
  # Output layer: 10 neurons (one for each digit 0-9) with softmax activation
  layer_dense(units = 10, activation = "softmax")

# Create the model by connecting inputs and outputs
model <- keras_model(inputs = inputs, outputs = outputs)


# Display model architecture summary
# This shows the structure, number of parameters, and output shapes of each layer
cat("Model Architecture Summary:\n")
model$summary()

#########################################################################
## Model Compilation
#########################################################################

# Before training, we need to configure the model with:
# 1. Optimizer: algorithm to update weights during training
# 2. Loss function: measures how well the model is performing
# 3. Metrics: additional measures to track during training

model$compile(
  optimizer = "adam",                  # Adam optimizer: adaptive learning rate, good default choice
  loss = "categorical_crossentropy",   # Loss function for multi-class classification with one-hot labels
  metrics = list("accuracy")           # Track accuracy during training
)



#########################################################################
## Model Training
#########################################################################

# Now we train the model on our data
# This is where the neural network learns to recognize handwritten digits
# Training involves:
# 1. Forward pass: predict digit from image
# 2. Calculate loss: how wrong the prediction is
# 3. Backward pass: adjust weights to improve predictions
# 4. Repeat for all training data

cat("Starting model training...\n")
cat("This may take a few minutes depending on your computer.\n")

history <- model$fit(
  x_train, y_train,                    # Training data and labels
  epochs = 5L,                         # Number of complete passes through the training data
  batch_size = 32L,                    # Number of samples processed before updating weights
  validation_data = list(x_test, y_test) # Test data to monitor performance during training
)

cat("Training completed!\n")


#########################################################################
## Model Evaluation
#########################################################################

# Evaluate the trained model on the test set
# This gives us the final performance metrics on unseen data
cat("Evaluating model on test set:\n")
test_results <- model$evaluate(x_test, y_test)
cat("Test loss:", test_results[1], "\n")
cat("Test accuracy:", test_results[2], "\n")


#########################################################################
## Training History Visualization
#########################################################################

# Plot the training history to understand how the model learned
# This helps us see if the model is learning properly and if there's overfitting

# Convert Python history object to R data frame
hist_list <- py_to_r(history$history)   # Contains: loss, accuracy, val_loss, val_accuracy
df <- as.data.frame(hist_list)

# Set up plotting parameters
par(mfrow = c(1, 1))


# Plot 1: Training and Validation Loss
# Loss measures how wrong the model's predictions are
# We want to see the loss decreasing over time
cat("Plotting training and validation loss:\n")
plot(df$loss, type = "o", pch = 16, col = "blue",
     xlab = "Epoch", ylab = "Loss", ylim = range(c(df$loss, df$val_loss)),
     main = "Training and Validation Loss")
lines(df$val_loss, type = "o", pch = 16, col = "red")
legend("topright", legend = c("Train Loss", "Validation Loss"),
       col = c("blue", "red"), lty = 1, pch = 16)

# Add numerical values to the plot for precise reading
text(x = 1:length(df$loss), y = df$loss, labels = round(df$loss, 4), pos = 3, col = "blue", cex = 0.8)
text(x = 1:length(df$val_loss), y = df$val_loss, labels = round(df$val_loss, 4), pos = 3, col = "red", cex = 0.8)


# Plot 2: Training and Validation Accuracy
# Accuracy measures the percentage of correct predictions
# We want to see accuracy increasing over time
cat("Plotting training and validation accuracy:\n")
plot(df$accuracy, type = "o", pch = 16, col = "blue",
     xlab = "Epoch", ylab = "Accuracy", ylim = range(c(df$accuracy, df$val_accuracy)),
     main = "Training and Validation Accuracy")
lines(df$val_accuracy, type = "o", pch = 16, col = "red")
legend("bottomright", legend = c("Train Accuracy", "Validation Accuracy"),
       col = c("blue", "red"), lty = 1, pch = 16)

# Add numerical values to the plot for precise reading
text(x = 1:length(df$accuracy), y = df$accuracy, labels = round(df$accuracy, 4), pos = 3, col = "blue", cex = 0.8)
text(x = 1:length(df$val_accuracy), y = df$val_accuracy, labels = round(df$val_accuracy, 4), pos = 3, col = "red", cex = 0.8)



#########################################################################
## Confusion Matrix Analysis
#########################################################################

# A confusion matrix shows how well our model performs on each digit class
# It helps us understand which digits are easier or harder to classify

# Step 1: Get model predictions on test set
# The model outputs probabilities for each digit (0-9)
y_pred <- model$predict(x_test)

# Step 2: Convert probabilities to predicted class labels
# max.col() finds the column with highest probability for each image
# We subtract 1 because R uses 1-based indexing but digits are 0-9
y_pred_classes <- max.col(y_pred) - 1

# Step 3: Convert one-hot encoded test labels back to class labels
# Same process: find the position of 1 in each row and subtract 1
y_true <- max.col(y_test) - 1

# Now both y_pred_classes and y_true contain values 0-9 representing the actual digits




# Create and visualize the confusion matrix
# The confusion matrix shows:
# - Rows: True digit labels
# - Columns: Predicted digit labels
# - Values: Number of predictions
# - Diagonal: Correct predictions (should be high)
# - Off-diagonal: Incorrect predictions (should be low)

cat("Creating confusion matrix visualization:\n")
{
  # Create labels and confusion matrix
  labels <- 0:9
  tab <- table(factor(y_true, levels = labels),
               factor(y_pred_classes, levels = labels))
  conf_mat <- as.matrix(tab)
  dimnames(conf_mat) <- list(True = labels, Pred = labels)
  
  # Flip rows for correct image() orientation
  z <- t(conf_mat[rev(seq_len(nrow(conf_mat))), , drop = FALSE])
  
  # Color palette: white (low) to dark blue (high)
  cols <- colorRampPalette(c("white", "lightblue", "blue", "darkblue"))(100)
  
  # Plot heatmap
  par(mfrow = c(1,1), mar = c(5,5,4,2) + 0.1)
  image(x = 1:ncol(conf_mat), y = 1:nrow(conf_mat), z = z,
        col = cols, axes = FALSE,
        xlab = "Predicted Label", ylab = "True Label",
        main = "Confusion Matrix: MNIST Digit Classification")
  
  # Add axis labels
  axis(1, at = 1:10, labels = labels)
  axis(2, at = 1:10, labels = rev(labels))
  
  # Overlay the actual counts on each cell
  for (i in 1:10) {
    for (j in 1:10) {
      true_idx <- 11 - j
      pred_idx <- i
      text(i, j, labels = conf_mat[true_idx, pred_idx], cex = 0.9)
    }
  }
}

# Interpretation:
# - High values on the diagonal (top-left to bottom-right) = good performance
# - High values off the diagonal = confusion between digits
# - Look for patterns: which digits are most commonly confused?





#########################################################################
## Part I Summary and Next Steps
#########################################################################

# Congratulations! You've successfully:
# 1. Set up TensorFlow/Keras in R
# 2. Loaded and preprocessed the MNIST dataset
# 3. Built and trained a CNN for digit recognition
# 4. Evaluated the model performance
# 5. Visualized the results

# Key takeaways from Part I:
# - CNNs are effective for image classification
# - Data preprocessing is crucial (normalization, reshaping)
# - Training history plots help diagnose learning issues
# - Confusion matrices reveal class-specific performance

# What to try next:
# - Experiment with different architectures (more/fewer layers)
# - Try different optimizers (SGD, RMSprop)
# - Adjust learning rates and batch sizes
# - Add dropout layers to prevent overfitting

#########################################################################
## Troubleshooting Guide
#########################################################################

# Common issues and solutions:

# 1. "Python not found" error:
#    - Install Python 3.x from python.org
#    - Make sure Python is in your system PATH

# 2. "TensorFlow installation failed":
#    - Try: install_tensorflow()
#    - For Apple Silicon Macs, use the commented code above

# 3. "Out of memory" error:
#    - Reduce batch_size (e.g., from 32 to 16)
#    - Close other applications
#    - Use a smaller subset of data for testing

# 4. "Model training is very slow":
#    - Check if you have a GPU available
#    - Reduce the number of epochs for testing
#    - Use a smaller model architecture

# 5. "Poor model performance":
#    - Increase the number of epochs
#    - Try a more complex architecture
#    - Check if data preprocessing is correct

#########################################################################
## END of PRACTICAL PART I
#########################################################################
