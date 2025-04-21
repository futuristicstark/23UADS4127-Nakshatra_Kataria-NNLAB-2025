# Experiment: CNN on Fashion MNIST with Keras

## Objective

Write a Python program to train and evaluate a Convolutional Neural Network (CNN) using the **Keras** library to classify the **Fashion MNIST** dataset. Demonstrate the effect of **filter size**, **regularization**, **batch size**, and **optimization algorithm** on model performance.

---

## Description of the Model

This experiment implements a **Convolutional Neural Network (CNN)** using the **Keras API** to classify images from the **Fashion MNIST** dataset, which consists of grayscale images representing 10 different clothing categories.

The model architecture includes:

- **Two convolutional layers** with customizable filter sizes (3×3 or 5×5), enabling experimentation with feature extraction.
- **MaxPooling layers** after each convolution layer to reduce dimensionality and retain important features.
- A **Flatten layer** followed by a **Dense (fully connected) layer** to interpret extracted features.
- A final **Dense layer with softmax activation** to classify input images into 10 classes.

### Additional Features:

- **L2 Regularization** is applied to convolutional and dense layers to minimize overfitting by penalizing large weights.
- The model supports multiple **optimizers**: `Adam`, `SGD`, and `RMSprop`, allowing comparison of learning dynamics.
- **Filter size**, **regularization strength**, **batch size**, and **optimizer** are treated as hyperparameters to observe their effect on model accuracy and convergence.

Each configuration is trained for **25 epochs**, and both training and validation accuracy are plotted to analyze the performance.

---

## Description of Code

- **Data Preprocessing**: 
  - The Fashion MNIST dataset is loaded, normalized (scaled between 0 and 1), and reshaped to (28, 28, 1) for CNN compatibility.

- **Model Building**:
  - The `build_model()` function creates a CNN using configurable filter sizes and L2 regularization. The final layer uses softmax for multiclass classification.
  
- **Training and Evaluation**:
  - The `train_and_evaluate()` function compiles the model using the selected optimizer and trains it on the training data.
  - After training, the model is evaluated on the test set to get final accuracy.

- **Visualization**:
  - Accuracy curves are plotted for each experiment using `matplotlib` to compare the impact of hyperparameter variations.

---

## Hyperparameter Experiments

The model is trained using the following configurations:

1. Filter Size: `3x3`, Regularization: `0.001`, Batch Size: `32`, Optimizer: `Adam`
2. Filter Size: `5x5`, Regularization: `0.0005`, Batch Size: `64`, Optimizer: `SGD`
3. Filter Size: `3x3`, Regularization: `0.0001`, Batch Size: `128`, Optimizer: `RMSprop`

For each setting, training/validation accuracy is plotted and test accuracy is printed to assess the configuration’s performance.

---

## Conclusion

This experiment demonstrates how **hyperparameters like filter size, regularization strength, batch size, and optimizer** significantly influence the training efficiency and generalization performance of CNNs on the Fashion MNIST dataset. Visualizing and comparing training histories helps in tuning models for better performance.
