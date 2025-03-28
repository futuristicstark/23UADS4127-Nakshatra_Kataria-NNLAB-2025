import tensorflow as tf
import numpy as np

# Load MNIST dataset
(x_train_set, y_train_set), (x_test_set, y_test_set) = tf.keras.datasets.mnist.load_data()
x_train_set = x_train_set.reshape(-1, 784).astype("float32") / 255.0
x_test_set = x_test_set.reshape(-1, 784).astype("float32") / 255.0

# Define placeholders
input_data = tf.placeholder(tf.float32, [None, 784])
target_output = tf.placeholder(tf.float32, [None, 10])

# Neural network architecture
layer1_size = 256
weights_1 = tf.Variable(tf.random.normal([784, layer1_size]))
bias_1 = tf.Variable(tf.zeros([layer1_size]))

weights_2 = tf.Variable(tf.random.normal([layer1_size, 10]))
bias_2 = tf.Variable(tf.zeros([10]))

# Forward pass
layer1_output = tf.nn.relu(tf.matmul(input_data, weights_1) + bias_1)
final_output = tf.matmul(layer1_output, weights_2) + bias_2
predicted_output = tf.nn.softmax(final_output)

# Loss and optimizer
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_output, logits=final_output))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

# Accuracy calculation
correct_predictions = tf.equal(tf.argmax(predicted_output, 1), tf.argmax(target_output, 1))
accuracy_metric = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Training
num_epochs = 10
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        for i in range(0, len(x_train_set), batch_size):
            batch_input = x_train_set[i:i + batch_size]
            batch_target = y_train_set[i:i + batch_size]
            batch_target = np.eye(10)[batch_target]

            sess.run(train_step, feed_dict={input_data: batch_input, target_output: batch_target})

        # Evaluate accuracy after each epoch
        train_accuracy = sess.run(accuracy_metric, feed_dict={input_data: x_train_set, target_output: np.eye(10)[y_train_set]})
        print(f"Epoch: {epoch + 1}, Training Accuracy: {train_accuracy:.4f}")

    # Test accuracy
    test_accuracy = sess.run(accuracy_metric, feed_dict={input_data: x_test_set, target_output: np.eye(10)[y_test_set]})
    print(f"Test Accuracy: {test_accuracy:.4f}")
