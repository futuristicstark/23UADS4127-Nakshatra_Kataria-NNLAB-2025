# 2. WAP to implement a multi-layer perceptron (MLP) network with one hidden layer using numpy in Python. Demonstrate that it can learn the XOR Boolean function.

import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_units, output_dim, lr=0.1, max_epochs=10000):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.lr = lr
        self.max_epochs = max_epochs

        self.W1 = np.random.randn(input_dim, hidden_units)
        self.b1 = np.zeros((1, hidden_units))
        self.W2 = np.random.randn(hidden_units, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation(self.z2)
        return self.a2

    def backward_pass(self, X, y):
        error_out = y - self.a2
        delta_out = error_out * self.activation_derivative(self.a2)

        error_hidden = delta_out.dot(self.W2.T)
        delta_hidden = error_hidden * self.activation_derivative(self.a1)

        self.W2 += self.a1.T.dot(delta_out) * self.lr
        self.b2 += np.sum(delta_out, axis=0, keepdims=True) * self.lr
        self.W1 += X.T.dot(delta_hidden) * self.lr
        self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * self.lr

    def train(self, X, y):
        for epoch in range(self.max_epochs):
            self.forward_pass(X)
            self.backward_pass(X, y)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.a2))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward_pass(X)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_dim=2, hidden_units=4, output_dim=1, lr=0.1, max_epochs=10000)
nn.train(X, y)

print("\nPredictions after training:")
print(nn.predict(X))
