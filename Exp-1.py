import numpy as np

class SimpleNeuron:
    def __init__(self, num_inputs, lr=0.1, max_iterations=1000):
        self.params = np.zeros(num_inputs + 1)
        self.lr = lr
        self.max_iterations = max_iterations

    def step_function(self, value):
        return 1 if value >= 0 else 0

    def compute_output(self, data_point):
        result = np.dot(data_point, self.params[1:]) + self.params[0]
        return self.step_function(result)

    def fit(self, dataset, targets):
        for _ in range(self.max_iterations):
            for sample, target in zip(dataset, targets):
                prediction = self.compute_output(sample)
                error = target - prediction
                self.params[1:] += self.lr * error * sample
                self.params[0] += self.lr * error


nand_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_target = np.array([1, 1, 1, 0])

xor_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_target = np.array([0, 1, 1, 0])

print("Training SimpleNeuron for NAND gate...")
nand_neuron = SimpleNeuron(num_inputs=2)
nand_neuron.fit(nand_data, nand_target)

print("Testing SimpleNeuron on NAND gate...")
for sample in nand_data:
    print(f"Input: {sample}, Predicted: {nand_neuron.compute_output(sample)}")

print("\nTraining SimpleNeuron for XOR gate...")
xor_neuron = SimpleNeuron(num_inputs=2)
xor_neuron.fit(xor_data, xor_target)

print("Testing SimpleNeuron on XOR gate...")
for sample in xor_data:
    print(f"Input: {sample}, Predicted: {xor_neuron.compute_output(sample)}")


