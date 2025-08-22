
# Course: Machine Learning for Health Research

#########################################################################
#########################################################################

## Practical III: Neural Networks - Part II

#########################################################################

### LEARNING OBJECTIVES FOR PART II
# By the end of this practical, you will be able to:
# 1. Work with real medical imaging data (MRI brain scans)
# 2. Handle multi-class classification (3 tumor types)
# 3. Implement data augmentation and preprocessing for medical images
# 4. Build a CNN for medical image classification
# 5. Use early stopping to prevent overfitting
# 6. Interpret results in a medical context

### DATASET OVERVIEW
# This practical uses MRI brain images from a publicly available dataset
# Three classes of brain conditions:
# - No tumor: Normal brain tissue
# - Glioma: A type of brain tumor that starts in glial cells
# - Meningioma: A tumor that forms in the membranes surrounding the brain

### MEDICAL CONTEXT
# Brain tumor classification from MRI images is crucial for:
# - Early diagnosis and treatment planning
# - Distinguishing between different tumor types
# - Monitoring treatment response
# - Research and clinical decision support

### IMPORTANT NOTES:
# - This dataset is larger than MNIST - ensure sufficient memory
# - Training will take longer than Part I
# - Medical image analysis requires careful validation
# - Results should be interpreted with clinical expertise

#########################################################################
## Setup: Installing and Loading Required Packages
#########################################################################

# This section installs and loads all necessary packages for deep learning
# reticulate: Interface between R and Python
# tensorflow: Deep learning framework
# keras: High-level neural network API
# magrittr: Provides the %>% pipe operator for cleaner code
# abind: For combining arrays (used for batch processing)
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
## Part II: MRI Brain Tumor Classification
#########################################################################

# This practical uses real medical imaging data from a publicly available dataset
# The dataset contains MRI brain images with three categories:
# - No tumor: Normal brain tissue
# - Glioma: A type of brain tumor that starts in glial cells
# - Meningioma: A tumor that forms in the membranes surrounding the brain

# Medical Context:
# Brain tumor classification from MRI images is crucial for:
# - Early diagnosis and treatment planning
# - Distinguishing between different tumor types
# - Monitoring treatment response
# - Research and clinical decision support


#########################################################################
## Dataset Exploration and Analysis
#########################################################################

# First, let's explore the dataset structure and understand what we're working with
# This helps us verify the data is properly organized and understand the class distribution

# Function to count images in a single folder
# This helps us verify that images are present in each category
count_images_in_folder <- function(folder_path) {
  files <- list.files(folder_path)  # Get all file names in the folder
  num_images <- sum(grepl("\\.jpg$", files, ignore.case = TRUE))  # Count only .jpg files
  return(num_images)
}

# Function to count images in all subfolders
# This gives us a complete overview of the dataset structure
count_images_in_all_folders <- function(parent_folder) {
  subfolders <- list.dirs(parent_folder, full.names = TRUE, recursive = FALSE)
  
  total_images <- 0
  for (folder in subfolders) {
    num_images <- count_images_in_folder(folder)
    total_images <- total_images + num_images
    cat(sprintf("Number of images in '%s': %d\n", folder, num_images))
  }
  
  return(total_images)
}


# Analyze the training dataset
# This shows us how many images we have in each category for training
cat("Analyzing training dataset:\n")
parent_folder <- "Training"
total_images <- count_images_in_all_folders(parent_folder)
cat("\nTotal number of images across all folders of Training set:", total_images, "\n")

# Analyze the testing dataset
# This shows us how many images we have in each category for testing
cat("\nAnalyzing testing dataset:\n")
parent_folder = 'Testing'
total_images = count_images_in_all_folders(parent_folder)
cat("\nTotal number of images across all folders of Test set:", total_images, "\n")





#########################################################################
## Data Visualization and Quality Check
#########################################################################

# Before training, it's important to visualize the data to understand:
# - What the images look like
# - Image quality and consistency
# - Potential issues with the dataset

# Load the jpeg package for reading image files
if (!require(jpeg, quietly = TRUE)) {
  install.packages("jpeg")
}
library(jpeg)

# Function to visualize multiple images in a grid
# This helps us understand the visual characteristics of each class
visualize_images_in_grid <- function(image_folder, grid_size = c(3, 3), num_images = 9) {
  # List all .jpg files in the folder
  image_files <- list.files(image_folder, pattern = "\\.jpg$", ignore.case = TRUE)
  if (length(image_files) == 0) {
    message("No .jpg files found in: ", normalizePath(image_folder, mustWork = FALSE))
    return(invisible(NULL))
  }
  
  # Sort and select the first N images
  selected_images <- head(sort(image_files), num_images)
  
  # Set up the plotting grid
  rows <- grid_size[1]; cols <- grid_size[2]
  par(mfrow = c(rows, cols), mar = c(1, 1, 2, 1))
  
  # Display each image
  for (fname in selected_images) {
    path <- file.path(image_folder, fname)
    img <- try(readJPEG(path), silent = TRUE)
    if (inherits(img, "try-error")) {
      plot.new(); title(fname, cex.main = 0.7); next
    }
    
    # Display the image (img is 2D for grayscale or 3D for RGB with values in [0,1])
    plot(0:1, 0:1, type = "n", xlab = "", ylab = "", axes = FALSE, main = fname, cex.main = 0.7)
    rasterImage(img, 0, 0, 1, 1)
  }
}

# Visualize sample images from the "no tumor" class
# This helps us understand what normal brain tissue looks like in the dataset
cat("Visualizing sample images from 'no tumor' class:\n")
image_folder_2 <- "Training/notumor"   # Adjust path if needed
visualize_images_in_grid(image_folder_2, grid_size = c(3,3), num_images = 9)

# Data Quality Note:
# You may notice various MRI sequences (T1, T2) and even CT scans in the dataset
# This is common in real-world medical imaging datasets
# A thorough data quality check is necessary before clinical application!



# Check the dimensions of a sample image
# This helps us understand the image format and size
cat("Checking image dimensions:\n")
img <- readJPEG("Training/notumor/Tr-noTr_0007.jpg")
cat("Image dimensions:", dim(img), "\n")
# The output shows: [height, width, channels] for RGB images
# or [height, width] for grayscale images






#########################################################################
## Data Preprocessing for Medical Images
#########################################################################

# Medical images often come in different sizes and formats
# We need to standardize them for neural network training
# This involves resizing all images to a consistent size

# Define preprocessing parameters
train_dir <- "Training"    # Directory containing training images
test_dir  <- "Testing"     # Directory containing test images
img_size  <- c(128, 128)   # Target image size (height, width)
batch_sz  <- 32L           # Number of images processed together

# Create training dataset
# This automatically loads images, resizes them, and creates labels
cat("Creating training dataset...\n")
train_ds <- image_dataset_from_directory(
  directory   = train_dir,      # Path to training images
  labels      = "inferred",     # Labels from folder names
  label_mode  = "categorical",  # One-hot encoding for multi-class
  image_size  = img_size,       # Resize all images to 128x128
  batch_size  = batch_sz,       # Process 32 images at a time
  shuffle     = TRUE,           # Randomize order for training
  seed        = 123             # For reproducibility
)

# Create test dataset
# Similar to training but without shuffling (order doesn't matter for testing)
cat("Creating test dataset...\n")
test_ds <- image_dataset_from_directory(
  directory   = test_dir,       # Path to test images
  labels      = "inferred",     # Labels from folder names
  label_mode  = "categorical",  # One-hot encoding for multi-class
  image_size  = img_size,       # Resize all images to 128x128
  batch_size  = batch_sz,       # Process 32 images at a time
  shuffle     = FALSE           # No shuffling needed for testing
)

# Function to convert TensorFlow dataset to R arrays
# This allows us to work with the data more easily in R
dataset_to_arrays <- function(dataset) {
  batches <- as_iterator(dataset)  # Convert to iterator
  imgs <- list()  # Store images
  labs <- list()  # Store labels
  
  # Process each batch
  repeat {
    batch <- tryCatch(iter_next(batches), error = function(e) NULL)
    if (is.null(batch)) break  # End of dataset
    imgs[[length(imgs) + 1]] <- as.array(batch[[1]])  # Images
    labs[[length(labs) + 1]] <- as.array(batch[[2]])  # Labels
  }
  
  # Combine all batches into single arrays
  x <- abind::abind(imgs, along = 1)  # Stack images
  y <- abind::abind(labs, along = 1)  # Stack labels
  list(x = x, y = y)
}

# Load abind package for combining arrays
# This is needed to merge batches into single arrays
if (!require(abind, quietly = TRUE)) {
  install.packages("abind")
}
library(abind)

# Convert datasets to arrays for easier manipulation
cat("Converting training dataset to arrays...\n")
train_arrays <- dataset_to_arrays(train_ds)

cat("Converting test dataset to arrays...\n")
test_arrays  <- dataset_to_arrays(test_ds)

# Extract images and labels
x_train_mri <- train_arrays$x  # Training images
y_train_mri <- train_arrays$y  # Training labels
x_test_mri  <- test_arrays$x   # Test images
y_test_mri  <- test_arrays$y   # Test labels



# Verify the data shapes
cat("Data shapes after preprocessing:\n")
cat("x_train_mri shape:", dim(x_train_mri), "\n")  # Should be (samples, 128, 128, 3)
cat("y_train_mri shape:", dim(y_train_mri), "\n")  # Should be (samples, 3)
cat("x_test_mri shape:", dim(x_test_mri), "\n")    # Should be (samples, 128, 128, 3)
cat("y_test_mri shape:", dim(y_test_mri), "\n")    # Should be (samples, 3)

# Understanding the dimensions:
# - 128 x 128: Image size (height x width)
# - 3: Number of color channels (RGB - Red, Green, Blue)
# - This is the standard format for color images in deep learning



# Final preprocessing step: Normalize pixel values
# Convert pixel values from 0-255 range to 0-1 range
# This helps neural networks train more effectively
cat("Normalizing pixel values...\n")
x_train_mri <- x_train_mri / 255  # Divide by 255 to get values in [0,1]
x_test_mri  <- x_test_mri / 255




#########################################################################
## Building the CNN for Medical Image Classification
#########################################################################

# Now we'll build a more sophisticated CNN for medical image classification
# This architecture is designed to handle the complexity of medical images
# and the three-class classification problem

# Define the input layer for 128x128 RGB images
inputs <- layer_input(shape = c(128, 128, 3))

# Build the CNN architecture layer by layer
outputs <- inputs %>%
  # First convolutional block: learns basic features (edges, textures)
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Second convolutional block: learns more complex features
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Third convolutional block: learns high-level features
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Flatten: convert 2D feature maps to 1D vector
  layer_flatten() %>%
  
  # Dense layers for classification
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%  # Prevent overfitting
  
  # Output layer: 3 neurons (one for each class) with softmax activation
  layer_dense(units = 3, activation = "softmax")

# Create the model
model_mri <- keras_model(inputs = inputs, outputs = outputs)


# Display model architecture summary
# This shows the structure, number of parameters, and output shapes of each layer
cat("Model Architecture Summary:\n")
model_mri$summary()

#########################################################################
## Model Compilation
#########################################################################

# Configure the model for training
# This sets up the optimizer, loss function, and metrics
model_mri$compile(
  optimizer = "adam",                  # Adam optimizer: adaptive learning rate
  loss = "categorical_crossentropy",   # Loss function for multi-class classification
  metrics = list("accuracy")           # Track accuracy during training
)





#########################################################################
## Model Training with Early Stopping
#########################################################################

# Set up early stopping callback
# This prevents overfitting by stopping training when performance stops improving
early_stopping <- callback_early_stopping(
  monitor = "val_loss",       # Monitor validation loss
  patience = 3,               # Stop after 3 epochs without improvement
  restore_best_weights = TRUE # Keep the best model weights
)

# Training parameters
epochs <- 10L  # Maximum number of training epochs

# Train the model
# Early stopping is crucial for medical applications to prevent overfitting
cat("Starting model training with early stopping...\n")
cat("This may take 10-30 minutes depending on your computer.\n")

history_mri <- model_mri$fit(
  x_train_mri, y_train_mri,                    # Training data
  epochs = epochs,                             # Maximum epochs
  batch_size = 32L,                           # Batch size
  validation_data = list(x_test_mri, y_test_mri), # Validation data
  callbacks = list(early_stopping)            # Early stopping callback
)

cat("Training completed!\n")





#########################################################################
## Training History Visualization
#########################################################################

# Function to plot training history
# This helps us understand how the model learned and detect overfitting
plot_training_history <- function(history_py) {
  library(reticulate)
  h <- py_to_r(history_py$history)   # Convert Python dict -> R list
  df <- as.data.frame(h)             # Columns: accuracy, val_accuracy, loss, val_loss
  
  # Set up plotting layout: 1 row, 2 columns
  par(mfrow = c(1, 2), mar = c(5,4,3,1))
  
  # Plot 1: Accuracy over time
  plot(df$accuracy, type = "l", lwd = 2,
       xlab = "Epoch", ylab = "Accuracy", main = "Model Accuracy",
       ylim = range(c(df$accuracy, df$val_accuracy), na.rm = TRUE))
  lines(df$val_accuracy, lwd = 2, col = "red")
  legend("bottomright", c("Train", "Validation"), lty = 1, lwd = 2, col = c("black","red"))
  
  # Plot 2: Loss over time
  plot(df$loss, type = "l", lwd = 2,
       xlab = "Epoch", ylab = "Loss", main = "Model Loss",
       ylim = range(c(df$loss, df$val_loss), na.rm = TRUE))
  lines(df$val_loss, lwd = 2, col = "red")
  legend("topright", c("Train", "Validation"), lty = 1, lwd = 2, col = c("black","red"))
  
  # Reset plotting parameters
  par(mfrow = c(1,1))
}

# Plot the training history
# This shows us how the model performed during training
cat("Plotting training history...\n")
plot_training_history(history_mri)




#########################################################################
## Model Evaluation and Prediction
#########################################################################

# Now let's evaluate our trained model on the test set
# This gives us the final performance metrics

# Define class names for interpretation
class_names <- c("meningiomas", "gliomas", "notumor")

# Get model predictions on test set
cat("Making predictions on test set...\n")
y_pred_probs <- model_mri$predict(x_test_mri)  # Get probability predictions
y_pred_classes <- max.col(y_pred_probs) - 1L   # Convert to class labels (0, 1, 2)

# Convert true labels from one-hot to class labels
y_true_classes <- max.col(y_test_mri) - 1L     # Convert to class labels (0, 1, 2)




#########################################################################
## Confusion Matrix Analysis
#########################################################################

# Create and visualize the confusion matrix
# This shows how well our model performs on each tumor type
cat("Creating confusion matrix...\n")

{
  # Create confusion matrix
  tab <- table(factor(y_true_classes, levels = 0:2),
               factor(y_pred_classes, levels = 0:2))
  conf_mat <- as.matrix(tab)
  dimnames(conf_mat) <- list(True = class_names, Pred = class_names)
  
  # Prepare flipped matrix for correct orientation
  z <- t(conf_mat[rev(seq_len(nrow(conf_mat))), ])
  
  # Color palette: white (low) to dark blue (high)
  cols <- colorRampPalette(c("white","lightblue","blue","darkblue"))(100)
  
  # Plot heatmap
  par(mfrow = c(1,1), mar = c(5,6,4,2) + 0.1)
  image(x = 1:ncol(conf_mat), y = 1:nrow(conf_mat), z = z,
        col = cols, axes = FALSE,
        xlab = "Predicted Label", ylab = "True Label",
        main = "Confusion Matrix: Brain Tumor Classification")
  
  # Add axis labels
  axis(1, at = 1:3, labels = class_names)
  axis(2, at = 1:3, labels = rev(class_names))
  
  # Overlay counts with adaptive text color for readability
  for (i in 1:3) for (j in 1:3) {
    ri <- 4 - j; ci <- i
    val <- conf_mat[ri, ci]
    txt_col <- ifelse(val > max(conf_mat)/2, "white", "black")
    text(i, j, labels = val, col = txt_col, cex = 1)
  }
}

# Interpretation:
# - High values on diagonal = good performance
# - Off-diagonal values = misclassifications
# - Each row shows how well the model identifies each tumor type
# - Medical implications: false negatives vs false positives have different clinical consequences


#########################################################################
## Part II Summary and Medical Implications
#########################################################################

# Congratulations! You've successfully:
# 1. Worked with real medical imaging data (MRI brain scans)
# 2. Built a CNN for medical image classification
# 3. Implemented early stopping to prevent overfitting
# 4. Evaluated model performance on brain tumor classification
# 5. Analyzed results in a medical context

# Key takeaways from Part II:
# - Medical image analysis requires careful preprocessing and validation
# - Early stopping is crucial for preventing overfitting in medical applications
# - Confusion matrices reveal class-specific performance important for clinical use
# - Real-world medical datasets often contain mixed modalities and quality issues

# Medical Implications:
# - False negatives (missing tumors) can be life-threatening
# - False positives (false alarms) can cause unnecessary stress and procedures
# - Model performance varies by tumor type - some are easier to detect than others
# - Clinical validation is essential before deployment in medical settings

# What to try next:
# - Experiment with data augmentation techniques
# - Try transfer learning with pre-trained models (ResNet, VGG)
# - Implement cross-validation for more robust evaluation
# - Add attention mechanisms for interpretability
# - Explore ensemble methods for improved performance

#########################################################################
## Advanced Topics for Further Study
#########################################################################

# 1. Transfer Learning:
#    - Use pre-trained models like ResNet or VGG
#    - Fine-tune on medical data for better performance
#    - Requires less data and training time

# 2. Data Augmentation:
#    - Rotate, flip, and adjust brightness of images
#    - Helps prevent overfitting and improves generalization
#    - Especially important for medical datasets with limited samples

# 3. Model Interpretability:
#    - Grad-CAM for visualizing what the model "sees"
#    - Attention mechanisms for highlighting important regions
#    - Crucial for gaining clinical trust and regulatory approval

# 4. Clinical Validation:
#    - Multi-center studies with diverse patient populations
#    - Comparison with radiologist performance
#    - Regulatory compliance (FDA, CE marking)

#########################################################################
## Troubleshooting Guide for Part II
#########################################################################

# Common issues and solutions:

# 1. "Out of memory" error during training:
#    - Reduce batch_size (e.g., from 32 to 16 or 8)
#    - Reduce image size (e.g., from 128x128 to 64x64)
#    - Close other applications to free memory
#    - Use data generators instead of loading all data at once

# 2. "Training is very slow":
#    - Check if GPU is available and being used
#    - Reduce model complexity (fewer layers/filters)
#    - Use a smaller subset for initial testing
#    - Consider transfer learning with pre-trained models

# 3. "Poor model performance":
#    - Check class balance in your dataset
#    - Try data augmentation techniques
#    - Increase model complexity gradually
#    - Verify data preprocessing is correct
#    - Check for data quality issues

# 4. "Early stopping triggers too quickly":
#    - Increase patience parameter (e.g., from 3 to 5)
#    - Check if learning rate is appropriate
#    - Verify data is properly normalized

# 5. "Confusion matrix shows bias":
#    - Check for class imbalance in dataset
#    - Consider weighted loss functions
#    - Use stratified sampling for training/validation splits

#########################################################################
## END of PRACTICAL PART II
#########################################################################

# You have now completed both parts of the Neural Networks practical!
# You've learned how to:
# - Set up deep learning environments in R
# - Work with both simple (MNIST) and complex (medical) datasets
# - Build and train CNNs for image classification
# - Evaluate model performance and interpret results
# - Apply deep learning to real-world medical problems

# Remember: Medical AI requires rigorous validation and clinical expertise!

