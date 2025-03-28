import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess dataset
x_train, x_test = x_train.reshape(-1, 28*28).astype(np.float32) / 255.0, x_test.reshape(-1, 28*28).astype(np.float32) / 255.0
y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

# Network parameters
input_size = 784
hidden_size1 = 128
hidden_size2 = 64
output_size = 10
learning_rate = 0.001
epochs = 30
batch_size = 128

# Initialize weights and biases using He initialization
initializer = tf.keras.initializers.HeNormal()
weights = {
    "W1": tf.Variable(initializer([input_size, hidden_size1])),
    "W2": tf.Variable(initializer([hidden_size1, hidden_size2])),
    "W3": tf.Variable(initializer([hidden_size2, output_size])),
}
biases = {
    "b1": tf.Variable(tf.zeros([hidden_size1])),
    "b2": tf.Variable(tf.zeros([hidden_size2])),
    "b3": tf.Variable(tf.zeros([output_size])),
}

# Dropout rate
dropout_rate = 0.3

# Define feed-forward function
def forward_propagation(x, training=True):
    z1 = tf.matmul(x, weights["W1"]) + biases["b1"]
    a1 = tf.nn.relu(z1)
    a1 = tf.nn.dropout(a1, rate=dropout_rate) if training else a1
    
    z2 = tf.matmul(a1, weights["W2"]) + biases["b2"]
    a2 = tf.nn.relu(z2)
    a2 = tf.nn.dropout(a2, rate=dropout_rate) if training else a2
    
    z3 = tf.matmul(a2, weights["W3"]) + biases["b3"]
    return z3  # Keep as logits

# Loss function (cross-entropy)
def compute_loss(y_pred, y_true):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

# Training function with backpropagation
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = forward_propagation(x_batch)
        loss = compute_loss(logits, y_batch)
    
    gradients = tape.gradient(loss, list(weights.values()) + list(biases.values()))
    
    for i, key in enumerate(weights.keys()):
        weights[key].assign_sub(learning_rate * gradients[i])
    for i, key in enumerate(biases.keys()):
        biases[key].assign_sub(learning_rate * gradients[len(weights) + i])
    
    return loss

# Store loss for plotting
loss_history = []

# Training loop
num_batches = x_train.shape[0] // batch_size
for epoch in range(epochs):
    avg_loss = 0
    for i in range(num_batches):
        batch_x = x_train[i * batch_size:(i + 1) * batch_size]
        batch_y = y_train[i * batch_size:(i + 1) * batch_size]
        loss = train_step(batch_x, batch_y)
        avg_loss += loss / num_batches
    
    loss_history.append(avg_loss.numpy())
    print(f"Epoch {epoch+1}, Loss: {avg_loss.numpy():.4f}")

# Plot the loss curve
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Training Loss', color='blue')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
logits_test = forward_propagation(x_test, training=False)
y_pred_test = tf.nn.softmax(logits_test)

correct_predictions = tf.equal(tf.argmax(y_pred_test, axis=1), tf.argmax(y_test, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

print(f"Test Accuracy: {accuracy.numpy() * 100:.2f}%")

# Confusion Matrix
y_true_labels = tf.argmax(y_test, axis=1).numpy()
y_pred_labels = tf.argmax(y_pred_test, axis=1).numpy()

confusion_matrix = tf.math.confusion_matrix(labels=y_true_labels, predictions=y_pred_labels, num_classes=10)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
