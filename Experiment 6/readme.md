# Experiment 6: Train and Evaluate a Recurrent Neural Network using PyTorch to Predict the Next Value in a Time Series Dataset

---

## Objective

Write a Python program to train and evaluate a **Recurrent Neural Network (RNN)** using the **PyTorch** library to predict the next value in a sample time series dataset.

---

## Model Description

We used a simple **RNN** for sequence-to-one regression to predict the **next day's temperature** based on prior data.

- **Input size**: 1 (daily temperature)
- **Hidden size**: 50
- **Layers**: 1 RNN layer + 1 Linear output layer
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

---

## Code Description

### 1. Preprocessing

- Loaded the daily minimum temperature dataset.
- Normalized the data using **MinMaxScaler**.
- Created **30-day sequences** to predict the next day’s temperature.
- Split data into training and test sets (80-20 split).
- Converted arrays to **PyTorch tensors** for model training.

### 2. Model Definition

- Defined an `RNNModel` class using `torch.nn.RNN` with:
  - A single hidden layer
  - A fully connected layer to produce the final output

### 3. Training

- Trained the model for **200 epochs**
- Calculated training **loss (MSE)** and **accuracy** at each epoch
- Accuracy was defined as `100 - MAE * 100` for an intuitive scale

### 4. Evaluation

- Used the following metrics:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² Score**
- Plotted:
  - Training loss curve
  - Accuracy curve
  - True vs predicted values
  - Final evaluation metrics

---

## Performance Evaluation

### Metrics:

| Metric | Value   |
|--------|---------|
| MSE    | 4.8267  |
| RMSE   | 2.1970  |
| MAE    | 1.7287  |
| R²     | 0.7128  |

- The **R² score of 0.71** indicates a reasonably good fit, meaning the model is able to capture a significant portion of the variance in the data.
- There's still room for improvement, particularly in reducing absolute error, but the model is much better than simply predicting the mean.

---

## Prediction

The model also predicted the **next day's temperature** using the latest 30-day sequence:

Predicted Next Day Temperature: 14.86°C

