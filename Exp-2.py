# 2. WAP to implement a multi-layer perceptron (MLP) network with one hidden layer using numpy in Python. Demonstrate that it can learn the XOR Boolean function.

import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.W1 = np.random.randn(input_size, hidden_size) 
        self.b1 = np.zeros((1, hidden_size))              
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))  

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        output_error = y - self.a2
        output_delta = output_error * self.sigmoid_derivative(self.a2)

        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        self.W2 += self.a1.T.dot(output_delta) * self.learning_rate
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.W1 += X.T.dot(hidden_delta) * self.learning_rate
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y):
        for epoch in range(self.epochs):
            self.forward(X)
            self.backward(X, y) 
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.a2))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]]) 

input_size = 2   # XOR has 2 input features
hidden_size = 4  # Number of neurons in the hidden layer (can be tuned)
output_size = 1  # XOR has 1 output

# Initialize MLP network
mlp = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size, learning_rate=0.1, epochs=10000)

# Train the network
mlp.train(X, y)

# Test the network
print("\nPredictions after training:")
predictions = mlp.predict(X)
print(predictions)

'''
Explanation:
The network was trained for 10,000 epochs using the XOR dataset.
The loss gradually decreased as the network learned the XOR function.
The output predictions after training are close to [0, 1, 1, 0], which is the expected output for the XOR truth table.
Tuning Parameters:
You can experiment with the hidden_size (number of neurons in the hidden layer) and learning_rate to improve performance and convergence speed.
The epochs value controls how many times the entire dataset is passed through the network during training. If necessary, you can adjust this number to reach a lower loss.
'''
