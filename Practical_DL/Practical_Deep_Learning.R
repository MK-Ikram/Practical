

# Course: Machine Learning for Health Research




#########################################################################
#########################################################################

## Practical III: Neural Networks 

#########################################################################



### This practical contains two part
# Part I: neural network on MNIST data set
# Part II: neural network for health related data set using MRI images



#########################################################################
## Setup (robust installs + environment)
#########################################################################

pkgs <- c("reticulate","tensorflow","keras","magrittr","abind","jpeg")         # /\
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]                  # /\
if (length(to_install)) install.packages(to_install)                           # /\

library(reticulate)
library(tensorflow)                                                            # /\
library(keras)                                                                 # /\
library(magrittr)

# Use system python by default (or switch to a conda env if you prefer)
use_python(Sys.which("python3"), required = TRUE)

# macOS Apple Silicon (optional): create/use conda env with TF-metal
# reticulate::install_miniconda(); conda_create("r-tf"); use_condaenv("r-tf", required=TRUE)
# py_install(c("tensorflow-macos","tensorflow-metal","h5py","numpy<2"), pip=TRUE)

# Reproducibility: set seeds in R, NumPy, and TensorFlow
set.seed(123)                                                                  # /\
np <- import("numpy", convert = FALSE); np$random$seed(123L)                   # /\
tf$random$set_seed(123L)                                                       # /\

# Quick platform check (kept)
R.version$arch; Sys.info()[["machine"]]

#########################################################################
## TensorFlow / Keras smoke test
#########################################################################

tf$version$VERSION
tf$math$add(1L, 2L)$numpy()  

#########################################################################
## Part I: MNIST dataset
#########################################################################

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





#########################################################################
## END of PRACTICAL 
#########################################################################
