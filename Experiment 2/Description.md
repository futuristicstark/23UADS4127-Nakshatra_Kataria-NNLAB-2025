# Objective
To implement a Multi-Layer Perceptron (MLP) using a step activation function to learn the XOR function, which is not linearly separable.

# Description of the Model
  The model consists of one hidden layer with two neurons and one output neuron.
  
  It follows a 3-layer architecture:
    Input Layer: Takes two binary inputs.
    Hidden Layer (2 neurons):
      First neuron mimics a NAND gate
      Second neuron mimics an OR gate
    Output Layer (1 neuron): Learns an AND gate using the hidden layer’s outputs.
    
 The model is trained using the Perceptron Learning Algorithm, updating weights using gradient descent.

# Description of the Code

## 1.Step Function:

  Implements a binary threshold activation function (>=0 → 1, else → 0).
  
## 2.Perceptron Class:

  Initializes random weights and bias.
  Uses a training loop with weight updates using the perceptron learning rule.
  Implements forward propagation for predictions.
  
## 3.Training Hidden Layer:

First neuron learns the NAND function.
Second neuron learns the OR function.

## 4.Training Output Layer:

  Takes the hidden layer outputs as input.
  Learns an AND function to compute XOR.
  
## 5.Performance Calculation:

  Computes final predictions.
  Calculates accuracy by comparing predictions with true XOR labels.


# Performance Evaluation

Accuracy Calculation
The model achieves 100% accuracy (4/4 correct predictions) for XOR inputs.

# Confusion Matrix 

| Actual ↓  | Predicted 0 | Predicted 1 |
|-----------|------------|------------|
| 0         | 2          | 0          |
| 1         | 0          | 2          |

![Loss Curve](Experiment_2/conf.png)

All values are correctly classified.

#Loss Curve
Since we use a step function, no gradient-based loss is computed.



 # My Comments (Limitations & Scope for Improvement)
## Limitations
  Step activation limits learning → Works only for simple problems like XOR, but not for complex datasets.
  
  Manual architecture design → Hidden layer functions (NAND, OR) are manually chosen instead of learning them dynamically.
