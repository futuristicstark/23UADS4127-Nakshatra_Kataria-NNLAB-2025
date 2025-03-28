
# Experiment Report: Neural Network for Handwritten Digit Classification

## **Objective**
WAP to implement a three-layer neural network using Tensor flow library (only, no keras) to classify MNIST handwritten digits dataset. Demonstrate the implementation of feed-forward and back-propagation approaches. 

---

## **Description of the Code **
This code implements a neural network from scratch using TensorFlow for the MNIST digit classification task. The MNIST dataset contains grayscale images of handwritten digits (0–9) of size 28x28 pixels. The goal is to classify each image into one of the 10 digits.

---


## **Description of the Code**
1. **Data Preparation**:
   Loading Dataset:

   - The code loads the MNIST dataset using tf.keras.datasets.mnist.load_data() which returns training and test data.

   Reshaping and Normalizing:
   
   - The images are reshaped from (28, 28) to a flattened vector of size 784.
   - Pixel values are normalized to the range [0, 1] for better training performance.
   
   One-Hot Encoding:
   
   - The labels (digits) are converted to one-hot vectors using tf.one_hot() to make them compatible with the softmax output layer.

2. **Model Building**:
   - Network Architecture:

     - Input layer → 784 nodes (28x28 pixels).
         
     - Hidden layer 1 → 128 nodes, ReLU activation.
         
     - Hidden layer 2 → 64 nodes, ReLU activation.
         
     - Output layer → 10 nodes (for digits 0–9), raw logits used for classification

  - Weights Initialization:

      - Weights are initalized using He Normal Initialization (tf.keras.initializers.HeNormal) for better gradient flow with ReLU activation.

3. **Compilation and Training**:
   - Adam optimizer and sparse categorical cross-entropy loss are used.
   - The model is trained for 10 epochs with validation.

4. **Evaluation**:
   - Predictions:

      - Model generates logits from test data.
      
      - Predicted class = argmax(logits).

    - Accuracy Calculation:

      - Compares predicted classes with true labels.
      
      - Accuracy = Mean of correctly predicted samples.

    - Confusion Matrix:

        - A confusion matrix is plotted using seaborn to visualize classification performance.
         
        - Provides insights into misclassified digits and model behavior.
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
- The model has only two hidden layers with 128 and 64 neurons, which may not be sufficient to capture complex patterns in the data.
- More complex architectures (like CNNs) are typically more effective for image data.
- The model continues training even if the loss stops decreasing, leading to wasted compute time and possible overfitting.

### ➡️ **Scope of Improvement**
- Use Convolutional Neural Networks (CNNs)
- Experiment with different optimizers or learning rates for better convergence.
- Add Early Stopping
- Adding more neurons or hidden layers may allow the model to learn more complex patterns.

