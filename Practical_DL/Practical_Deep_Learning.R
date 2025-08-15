

# Course: Machine Learning for Health Research




#########################################################################
#########################################################################

## Practical III: Neural Networks 

#########################################################################



### This practical contains two part
# Part I: neural network on MNIST data set
# Part II: neural network for health related data set using MRI images



#########################################################################

# Installation instruction for Mac

install.packages(c("reticulate", "tensorflow", "keras"))

library(reticulate)

# Use whatever Python R is already linked to
use_python(Sys.which("python3"), required = TRUE)

# Install directly into that Python
py_install(c("tensorflow", "keras", "h5py"), pip = TRUE)

#__________________________________________________
R.version$arch
Sys.info()[["machine"]]
#__________________________________________________

#########################################################################

# Laod Tensorflow and Keras and then obtain MNIST data set

# Test TensorFlow and Keras
library(tensorflow)
tf$version$VERSION
tf$add(1L, 2L)$numpy()





### Part I: MNIST data set

mnist <- dataset_mnist()
str(mnist)



# Obtain data frames for features (x) and labels (y), both for train and test sets.

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test  <- mnist$test$x
y_test  <- mnist$test$y



# Check number of rows in train and test sets

nrow(x_train)  # number of training samples
nrow(x_test)   # number of test samples

# Check the dimensions of the images

dim(x_train)



# Function to plot 5 samples
plot_samples <- function(x, y) {
  par(mfrow = c(1, 5), mar = c(1, 1, 2, 1))  # layout: 1 row, 5 columns
  for (i in 1:5) {
    image(t(apply(x[i,,], 2, rev)), col = gray.colors(256), axes = FALSE, main = paste("Label:", y[i]))
  }
}

# Visualize 5 samples from the training set
plot_samples(x_train, y_train)


# Pre-processing steps in preparation for the neural network

# Normalize pixel values to [0,1]
x_train <- x_train / 255
x_test  <- x_test / 255

# Reshape to (samples, 28, 28, 1)
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test  <- array_reshape(x_test,  c(nrow(x_test),  28, 28, 1))

# Check shapes
cat("x_train shape:", dim(x_train), "\n")  # Should be 60000 28 28 1
cat("x_test shape:",  dim(x_test),  "\n")  # Should be 10000 28 28 1


# Convert labels to categorical one-hot vector layout
y_train <- tf$keras$utils$to_categorical(y_train, num_classes = 10L)
y_test  <- tf$keras$utils$to_categorical(y_test,  num_classes = 10L)


dim(y_train)  # should be 60000 10
dim(y_test)   # should be 10000 10




# Build the CNN model for MNIST data set
# Here you can try out different neural network architectures

library(magrittr)  # for %>%

inputs <- layer_input(shape = c(28, 28, 1))

outputs <- inputs %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model <- keras_model(inputs = inputs, outputs = outputs)


# Obtain summary of the architecture

model$summary()


# Configure training
model$compile(
  optimizer = "adam",                  # Optimizer for adjusting weights during training.
  loss = "categorical_crossentropy",   # Multi-class loss (use one-hot labels; else use 'sparse_categorical_crossentropy').
  metrics = list("accuracy")           # Metrics to report during training/evaluation.
)



# Train the model
history <- model$fit(
  x_train, y_train,                    # Training features and labels.
  epochs = 5L,                         # Number of full passes through the training data.
  batch_size = 32L,                    # Samples per gradient update.
  validation_data = list(x_test, y_test) # Evaluate on held-out data (test data)
)


# Finale valuation on test data
model$evaluate(x_test,y_test)


# Prepare training history for plotting/analysis
hist_list <- py_to_r(history$history)   # list(loss, accuracy, val_loss, val_accuracy)
df <- as.data.frame(hist_list)

par(mfrow = c(1, 1))


# Plot 1
# ---- Loss plot ----
plot(df$loss, type = "o", pch = 16, col = "blue",
     xlab = "Epoch", ylab = "Loss", ylim = range(c(df$loss, df$val_loss)))
lines(df$val_loss, type = "o", pch = 16, col = "red")
legend("topright", legend = c("Train Loss", "Test Loss"),
       col = c("blue", "red"), lty = 1, pch = 16)

# Annotate values
text(x = 1:length(df$loss), y = df$loss, labels = round(df$loss, 4), pos = 3, col = "blue", cex = 0.8)
text(x = 1:length(df$val_loss), y = df$val_loss, labels = round(df$val_loss, 4), pos = 3, col = "red", cex = 0.8)


# Plot 2
# ---- Accuracy plot ----
plot(df$accuracy, type = "o", pch = 16, col = "blue",
     xlab = "Epoch", ylab = "Accuracy", ylim = range(c(df$accuracy, df$val_accuracy)))
lines(df$val_accuracy, type = "o", pch = 16, col = "red")
legend("bottomright", legend = c("Train Accuracy", "Test Accuracy"),
       col = c("blue", "red"), lty = 1, pch = 16)

# Annotate values
text(x = 1:length(df$accuracy), y = df$accuracy, labels = round(df$accuracy, 4), pos = 3, col = "blue", cex = 0.8)
text(x = 1:length(df$val_accuracy), y = df$val_accuracy, labels = round(df$val_accuracy, 4), pos = 3, col = "red", cex = 0.8)



## Obtain confusion matrix

# Predict probabilities on the test set
y_pred <- model$predict(x_test)

# Convert probabilities -> class ids (0..9)
y_pred_classes <- max.col(y_pred) - 1

# If your y_test is one-hot, convert back to class ids
y_true <- max.col(y_test) - 1

# Above lines subtracting -1 shifts the range from 1..10 to 0..9, amtching the actual digit labels!




# Confusion matrix heatmap in one go
{
  # Create labels and confusion matrix
  labels <- 0:9
  tab <- table(factor(y_true, levels = labels),
               factor(y_pred_classes, levels = labels))
  conf_mat <- as.matrix(tab)
  dimnames(conf_mat) <- list(True = labels, Pred = labels)
  
  # Flip rows for correct image() orientation
  z <- t(conf_mat[rev(seq_len(nrow(conf_mat))), , drop = FALSE])
  
  # Color palette
  cols <- colorRampPalette(c("white", "lightblue", "blue", "darkblue"))(100)
  
  # Plot heatmap
  par(mfrow = c(1,1), mar = c(5,5,4,2) + 0.1)
  image(x = 1:ncol(conf_mat), y = 1:nrow(conf_mat), z = z,
        col = cols, axes = FALSE,
        xlab = "Predicted Label", ylab = "True Label",
        main = "Confusion Matrix")
  
  # Axes
  axis(1, at = 1:10, labels = labels)
  axis(2, at = 1:10, labels = rev(labels))
  
  # Overlay counts
  for (i in 1:10) {
    for (j in 1:10) {
      true_idx <- 11 - j
      pred_idx <- i
      text(i, j, labels = conf_mat[true_idx, pred_idx], cex = 0.9)
    }
  }
}









### Part II: Health-related data set: MRI brain images

# These images are from the freely available online Kaggle website.
# For this practical three categories of images are selected: No tumour, glioma and meningeoma


# MRI Images

# Function written to obtain numbers from folders

# Count .jpg images in a single folder
count_images_in_folder <- function(folder_path) {
  files <- list.files(folder_path)  # get file names
  num_images <- sum(grepl("\\.jpg$", files, ignore.case = TRUE))
  return(num_images)
}

# Count .jpg images in all subfolders
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


# Example usage to obtain number for Training folder
parent_folder <- "Training"
total_images <- count_images_in_all_folders(parent_folder)

cat("\nTotal number of images across all folders of Training set:", total_images, "\n")


# Simialrly specify parent folder of test sets
parent_folder = 'Testing'
total_images = count_images_in_all_folders(parent_folder)

cat("\nTotal number of images across all folders of Test set:", total_images, "\n")





# Obtain sample images from the folders for inspection

install.packages("jpeg")

# install.packages("jpeg")  # run once if needed
library(jpeg)

# Fucntion to obtain images
visualize_images_in_grid <- function(image_folder, grid_size = c(3, 3), num_images = 9) {
  # list .jpg files
  image_files <- list.files(image_folder, pattern = "\\.jpg$", ignore.case = TRUE)
  if (length(image_files) == 0) {
    message("No .jpg files found in: ", normalizePath(image_folder, mustWork = FALSE))
    return(invisible(NULL))
  }
  
  # sort + select
  selected_images <- head(sort(image_files), num_images)
  
  rows <- grid_size[1]; cols <- grid_size[2]
  par(mfrow = c(rows, cols), mar = c(1, 1, 2, 1))
  
  for (fname in selected_images) {
    path <- file.path(image_folder, fname)
    img <- try(readJPEG(path), silent = TRUE)
    if (inherits(img, "try-error")) {
      plot.new(); title(fname, cex.main = 0.7); next
    }
    
    # img is 2D (grayscale) or 3D (RGB) with values in [0,1]
    plot(0:1, 0:1, type = "n", xlab = "", ylab = "", axes = FALSE, main = fname, cex.main = 0.7)
    rasterImage(img, 0, 0, 1, 1)
  }
}

# Example usage, obtain 9 images
image_folder_2 <- "Training/notumor"   # adjust to your path
visualize_images_in_grid(image_folder_2, grid_size = c(3,3), num_images = 9)


# You will notice that there are various MRI sequences (T1, T2), but even a CT scan.
# So, a thorough data quality check is necessary!



# Just to check dimensions of an arbitrary image

img <- readJPEG("Training/notumor/Tr-noTr_0007.jpg")
dim(img)






# Pre-processing to obtain the Train and Test sets in an appropriate format
# rescale all iamges to 128 by 128

train_dir <- "Training"
test_dir  <- "Testing"
img_size  <- c(128, 128)
batch_sz  <- 32L

# Training set
train_ds <- image_dataset_from_directory(
  directory   = train_dir,
  labels      = "inferred",
  label_mode  = "categorical",
  image_size  = img_size,
  batch_size  = batch_sz,
  shuffle     = TRUE,
  seed        = 123
)

# Test set
test_ds <- image_dataset_from_directory(
  directory   = test_dir,
  labels      = "inferred",
  label_mode  = "categorical",
  image_size  = img_size,
  batch_size  = batch_sz,
  shuffle     = FALSE
)

# Convert dataset to arrays
dataset_to_arrays <- function(dataset) {
  batches <- as_iterator(dataset)
  imgs <- list()
  labs <- list()
  repeat {
    batch <- tryCatch(iter_next(batches), error = function(e) NULL)
    if (is.null(batch)) break
    imgs[[length(imgs) + 1]] <- as.array(batch[[1]])
    labs[[length(labs) + 1]] <- as.array(batch[[2]])
  }
  x <- abind::abind(imgs, along = 1)
  y <- abind::abind(labs, along = 1)
  list(x = x, y = y)
}

# Need abind for combining batches
install.packages("abind")  # run once if not installed
library(abind)

train_arrays <- dataset_to_arrays(train_ds)
test_arrays  <- dataset_to_arrays(test_ds)

x_train_mri <- train_arrays$x
y_train_mri <- train_arrays$y
x_test_mri  <- test_arrays$x
y_test_mri  <- test_arrays$y



# Shapes check
cat("x_train_mri shape:", dim(x_train_mri), "\n")
cat("y_train_mri shape:", dim(y_train_mri), "\n")
cat("x_test_mri shape:", dim(x_test_mri), "\n")
cat("y_test_mri shape:", dim(y_test_mri), "\n")


# So the dimensions of each image are 128 by 128 by 3
# 128 by 128 is what we had decided to do ourselves, 3 is for the number of channels, that is R,G,B colours



# Final pre-processing step

x_train_mri <- x_train_mri / 255
x_test_mri  <- x_test_mri / 255




# Input: 128x128 RGB
inputs <- layer_input(shape = c(128, 128, 3))

# CNN stack
outputs <- inputs %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 3, activation = "softmax")

model_mri <- keras_model(inputs = inputs, outputs = outputs)


# Summary
model_mri$summary()


# Compile 
model_mri$compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = list("accuracy")
)





# EarlyStopping callback
early_stopping <- callback_early_stopping(
  monitor = "val_loss",       # metric to monitor
  patience = 3,               # stop after 3 epochs of no improvement
  restore_best_weights = TRUE # restore weights from best epoch
)

# Number of epochs
epochs <- 10L

# Fit model with EarlyStopping
# Early stopping is important in case of building large models and limited resources

history_mri <- model_mri$fit(
  x_train_mri, y_train_mri,
  epochs = epochs,
  batch_size = 32L,
  validation_data = list(x_test_mri, y_test_mri),
  callbacks = list(early_stopping)
)





plot_training_history <- function(history_py) {
  library(reticulate)
  h <- py_to_r(history_py$history)   # convert Python dict -> R list
  df <- as.data.frame(h)             # columns: accuracy, val_accuracy, loss, val_loss
  
  par(mfrow = c(1, 2), mar = c(5,4,3,1))
  
  # Accuracy
  plot(df$accuracy, type = "l", lwd = 2,
       xlab = "Epoch", ylab = "Accuracy", main = "Model Accuracy",
       ylim = range(c(df$accuracy, df$val_accuracy), na.rm = TRUE))
  lines(df$val_accuracy, lwd = 2, col = "red")
  legend("bottomright", c("Train", "Test"), lty = 1, lwd = 2, col = c("black","red"))
  
  # Loss
  plot(df$loss, type = "l", lwd = 2,
       xlab = "Epoch", ylab = "Loss", main = "Model Loss",
       ylim = range(c(df$loss, df$val_loss), na.rm = TRUE))
  lines(df$val_loss, lwd = 2, col = "red")
  legend("topright", c("Train", "Test"), lty = 1, lwd = 2, col = c("black","red"))
  
  par(mfrow = c(1,1))
}

# Obtain plot
plot_training_history(history_mri)




# Class names three categories
class_names <- c("mengiomas", "gliomas", "notumor")

# Predict on the whole test array
y_pred_probs <- model_mri$predict(x_test_mri)
y_pred_classes <- max.col(y_pred_probs) - 1L      # 0..2

# True labels (one-hot -> class ids)
y_true_classes <- max.col(y_test_mri) - 1L        # 0..2




# Confusion matrix (3x3) for test data

{
tab <- table(factor(y_true_classes, levels = 0:2),
             factor(y_pred_classes, levels = 0:2))
conf_mat <- as.matrix(tab)
dimnames(conf_mat) <- list(True = class_names, Pred = class_names)

# Prepare flipped matrix for correct orientation
z <- t(conf_mat[rev(seq_len(nrow(conf_mat))), ])

# Color palette
cols <- colorRampPalette(c("white","lightblue","blue","darkblue"))(100)

# Plot heatmap
par(mfrow = c(1,1), mar = c(5,6,4,2) + 0.1)
image(x = 1:ncol(conf_mat), y = 1:nrow(conf_mat), z = z,
      col = cols, axes = FALSE,
      xlab = "Predicted Label", ylab = "True Label",
      main = "Confusion Matrix: CNN trained on train data and checked on Test data")

# Axes
axis(1, at = 1:3, labels = class_names)
axis(2, at = 1:3, labels = rev(class_names))

# Overlay counts with adaptive text color
for (i in 1:3) for (j in 1:3) {
  ri <- 4 - j; ci <- i
  val <- conf_mat[ri, ci]
  txt_col <- ifelse(val > max(conf_mat)/2, "white", "black")
  text(i, j, labels = val, col = txt_col, cex = 1)
}
}




#########################################################################
## END of PRACTICAL 
#########################################################################
