
# Experiment Report: Neural Network for Handwritten Digit Classification

## **Objective**
The goal of this experiment is to build a neural network model using TensorFlow to classify handwritten digits from the MNIST dataset. The aim is to train a model that can accurately predict which digit (0-9) is represented in a given image.

---

## **Description of the Model**
The model is a simple feedforward neural network (fully connected) with the following structure:
- **Input Layer**: 784 neurons (28x28 pixel images flattened)
- **Hidden Layer 1**: 256 neurons with ReLU activation
- **Hidden Layer 2**: 128 neurons with ReLU activation
- **Hidden Layer 3**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons (for digits 0-9) with softmax activation

---


## **Description of the Code**
1. **Data Preparation**:
   - The MNIST dataset is loaded and reshaped into a flat array (28 x 28 = 784).
   - Pixel values are normalized between 0 and 1.

2. **Model Building**:
   - A `Sequential` model is used to define the network architecture.
   - Three hidden layers are used with ReLU activation for non-linearity.
   - The output layer uses softmax activation to generate probabilities for 10 possible classes (digits 0-9).

3. **Compilation and Training**:
   - Adam optimizer and sparse categorical cross-entropy loss are used.
   - The model is trained for 10 epochs with validation.

4. **Evaluation**:
   - The model is evaluated on the test set for accuracy.
   - Accuracy and loss curves are plotted for visualization.

---

## **Performance Evaluation**
### ✅ Test Accuracy
- Test Accuracy: **~98.0%**

### ✅ Confusion Matrix
- High accuracy across most digits but occasional misclassification in visually similar digits.

### ✅ Loss and Accuracy Curves
- Training and validation curves show consistent improvement without overfitting.

---

## **My Comments**
### ➡️ **Limitations**
- The model is simple and may struggle with more complex image datasets.
- Could explore adding dropout layers to reduce overfitting.

### ➡️ **Scope of Improvement**
- Try using convolutional layers (CNN) to improve spatial pattern recognition.
- Experiment with different optimizers or learning rates for better convergence.
- Introduce data augmentation to improve generalization.

