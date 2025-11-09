import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, prange
import os

FILENAME = 'mnist_test.csv'
USE_KAGGLEHUB = True
LOCAL_FILENAME = "mnist_test.csv"
NUM_TRAIN_SAMPLES = 128

learning_rate = 0.001
epochs = 15
batch_size = 32
dropout_rate = 0.3
lambda_l2_weights = 0.001


@jit(nopython=True, parallel=True)
def jit_conv_forward(N, num_filters, H_out, W_out, kernel_size, input_image, weights, biases):
    output = np.zeros((N, num_filters, H_out, W_out))
    for i in prange(N):
        for f in range(num_filters):
            for h in range(H_out):
                for w in range(W_out):
                    patch = input_image[i, :, h:h + kernel_size, w:w + kernel_size]
                    output[i, f, h, w] = np.sum(patch * weights[f]) + biases[f]
    return output


@jit(nopython=True)
def jit_conv_backward_weights_inputs(dvalues, inputs, weights, kernel_size, num_filters, H_out, W_out, N, C_in):
    dweights = np.zeros_like(weights)
    dinputs = np.zeros_like(inputs)

    for i in range(N):
        for f in range(num_filters):
            for h in range(H_out):
                for w in range(W_out):
                    patch = inputs[i, :, h:h + kernel_size, w:w + kernel_size]
                    dvalue = dvalues[i, f, h, w]
                    dinputs[i, :, h:h + kernel_size, w:w + kernel_size] += weights[f] * dvalue
                    dweights[f] += patch * dvalue

    return dinputs, dweights


@jit(nopython=True, parallel=True)
def jit_pool_forward(N, C, H_out, W_out, pool_size, stride, input_map):
    output = np.zeros((N, C, H_out, W_out))
    max_indices = np.zeros((N, C, H_out, W_out, 2), dtype=np.int64)

    for i in prange(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride
                    patch = input_map[i, c, h_start:h_start + pool_size, w_start:w_start + pool_size]

                    max_val = np.max(patch)
                    output[i, c, h, w] = max_val

                    found = False
                    for ph in range(pool_size):
                        for pw in range(pool_size):
                            if patch[ph, pw] == max_val:
                                max_indices[i, c, h, w] = [h_start + ph, w_start + pw]
                                found = True
                                break
                        if found:
                            break
    return output, max_indices


@jit(nopython=True, parallel=True)
def jit_pool_backward(dvalues, dinputs, max_indices, N, C, H_out, W_out):
    for i in prange(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    error_value = dvalues[i, c, h, w]
                    h_idx, w_idx = max_indices[i, c, h, w]
                    dinputs[i, c, h_idx, w_idx] += error_value
    return dinputs


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weights_lambda_l2=0, bias_lambda_l2=0):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.weights_lambda_l2 = weights_lambda_l2
        self.bias_lambda_l2 = bias_lambda_l2
        self.inputs = None
        self.output = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weights_lambda_l2 > 0:
            self.dweights += 2 * self.weights_lambda_l2 * self.weights
        if self.bias_lambda_l2 > 0:
            self.dbiases += 2 * self.bias_lambda_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)

    def backward(self, dvalues):
        d_leaky_relu = np.where(self.inputs > 0, 1, self.alpha)
        self.dinputs = dvalues * d_leaky_relu


class Activation_Softmax:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Layer_Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.keep_probability = 1 - rate
        self.inputs = None
        self.output = None
        self.dinputs = None
        self.mask = None
        self._training_mode = True

    def forward(self, inputs):
        self.inputs = inputs

        if not self._training_mode:
            self.output = inputs.copy()
            return

        self.mask = np.random.binomial(1, self.keep_probability, size=inputs.shape) / self.keep_probability
        self.output = inputs * self.mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask

    def set_prediction_mode(self):
        self._training_mode = False

    def set_training_mode(self):
        self._training_mode = True


class Conv2D:
    def __init__(self, num_filters, in_channel, kernel_size):
        self.num_filters = num_filters
        self.in_channel = in_channel
        self.kernel_size = kernel_size

        scale = np.sqrt(2.0 / (in_channel * kernel_size * kernel_size))
        self.weights = np.random.randn(num_filters, in_channel, kernel_size, kernel_size) * scale
        self.biases = np.zeros(num_filters)

        self.inputs = None
        self.output = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        N, C_in, H_in, W_in = inputs.shape
        H_out = H_in - self.kernel_size + 1
        W_out = W_in - self.kernel_size + 1

        self.output = jit_conv_forward(N, self.num_filters, H_out, W_out, self.kernel_size, inputs, self.weights,
                                       self.biases)
        return self.output

    def backward(self, dvalues):
        N, C_in, H_in, W_in = self.inputs.shape
        _, _, H_out, W_out = dvalues.shape

        self.dinputs, self.dweights = jit_conv_backward_weights_inputs(dvalues, self.inputs, self.weights,
                                                                       self.kernel_size, self.num_filters, H_out, W_out,
                                                                       N, C_in)

        self.dbiases = np.zeros_like(self.biases)
        for f in range(self.num_filters):
            self.dbiases[f] = np.sum(dvalues[:, f, :, :])

        return self.dinputs


class MaxPool2D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.inputs = None
        self.output = None
        self.dinputs = None
        self.max_indices = None

    def forward(self, inputs):
        self.inputs = inputs
        N, C_in, H_in, W_in = inputs.shape
        H_out = int((H_in - self.pool_size) / self.stride) + 1
        W_out = int((W_in - self.pool_size) / self.stride) + 1

        self.output, self.max_indices = jit_pool_forward(N, C_in, H_out, W_out, self.pool_size, self.stride, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.zeros_like(self.inputs)
        N, C_in, H_out, W_out = dvalues.shape

        self.dinputs = jit_pool_backward(dvalues, self.dinputs, self.max_indices, N, C_in, H_out, W_out)
        return self.dinputs


class Flatten:
    def __init__(self):
        self.inputs_shape = None

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        N = inputs.shape[0]
        return inputs.reshape(N, -1)

    def backward(self, dvalues):
        return dvalues.reshape(self.inputs_shape)


class Loss:
    def calculate(self, output, y, layers):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        reg_loss = self.regularization_loss(layers)
        return data_loss + reg_loss

    def regularization_loss(self, layers):
        reg_loss = 0
        for layer in layers:
            if isinstance(layer, Layer_Dense):
                if layer.weights_lambda_l2 > 0:
                    reg_loss += layer.weights_lambda_l2 * np.sum(layer.weights * layer.weights)
                if layer.bias_lambda_l2 > 0:
                    reg_loss += layer.bias_lambda_l2 * np.sum(layer.biases * layer.biases)
        return 0.5 * reg_loss


class Loss_CategoricalCrossentropy(Loss):
    def __init__(self):
        self.dinputs = None

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        else:
            raise ValueError("Invalid shape for y_true")

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        num_outputs = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true_one_hot = np.eye(num_outputs)[y_true]
        elif len(y_true.shape) == 2:
            y_true_one_hot = y_true
        else:
            raise ValueError("Invalid shape for y_true")

        self.dinputs = -y_true_one_hot / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
        self.output = None
        self.dinputs = None

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.forward(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true_indices = np.argmax(y_true, axis=1)
        elif len(y_true.shape) == 1:
            y_true_indices = y_true
        else:
            raise ValueError("Invalid shape for y_true")

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true_indices] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.momentums = {}
        self.caches = {}

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        layer_id = id(layer)

        if layer_id not in self.momentums:
            self.momentums[layer_id] = {'weights': np.zeros_like(layer.weights), 'biases': np.zeros_like(layer.biases)}
            self.caches[layer_id] = {'weights': np.zeros_like(layer.weights), 'biases': np.zeros_like(layer.biases)}

        self.momentums[layer_id]['weights'] = self.beta_1 * self.momentums[layer_id]['weights'] + (
                    1 - self.beta_1) * layer.dweights
        self.momentums[layer_id]['biases'] = self.beta_1 * self.momentums[layer_id]['biases'] + (
                    1 - self.beta_1) * layer.dbiases

        t = self.iterations + 1
        momentum_corrected_weights = self.momentums[layer_id]['weights'] / (1 - self.beta_1 ** t)
        momentum_corrected_biases = self.momentums[layer_id]['biases'] / (1 - self.beta_1 ** t)

        self.caches[layer_id]['weights'] = self.beta_2 * self.caches[layer_id]['weights'] + (
                    1 - self.beta_2) * layer.dweights ** 2
        self.caches[layer_id]['biases'] = self.beta_2 * self.caches[layer_id]['biases'] + (
                    1 - self.beta_2) * layer.dbiases ** 2

        cache_corrected_weights = self.caches[layer_id]['weights'] / (1 - self.beta_2 ** t)
        cache_corrected_biases = self.caches[layer_id]['biases'] / (1 - self.beta_2 ** t)

        layer.weights -= self.current_learning_rate * momentum_corrected_weights / (
                    np.sqrt(cache_corrected_weights) + self.epsilon)
        layer.biases -= self.current_learning_rate * momentum_corrected_biases / (
                    np.sqrt(cache_corrected_biases) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


print("Loading MNIST data...")

if USE_KAGGLEHUB:
    try:
        import kagglehub

        print("Downloading MNIST dataset using kagglehub...")
        path = kagglehub.dataset_download("oddrationale/mnist-in-csv")
        print(f"Dataset downloaded to: {path}")

        import glob

        csv_files = glob.glob(os.path.join(path, "*.csv"))

        if not csv_files:
            raise FileNotFoundError("No CSV files found in downloaded dataset")

        print(f"Found CSV files: {[os.path.basename(f) for f in csv_files]}")

        test_files = [f for f in csv_files if 'test' in os.path.basename(f).lower()]
        if test_files:
            FILENAME = test_files[0]
        else:
            FILENAME = csv_files[0]

        print(f"Using file: {os.path.basename(FILENAME)}")

    except ImportError:
        print("kagglehub not installed. Install it with: pip install kagglehub")
        print(f"Falling back to local file: {LOCAL_FILENAME}")
        FILENAME = LOCAL_FILENAME
        USE_KAGGLEHUB = False
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print(f"Falling back to local file: {LOCAL_FILENAME}")
        FILENAME = LOCAL_FILENAME
        USE_KAGGLEHUB = False
else:
    FILENAME = LOCAL_FILENAME

if not os.path.exists(FILENAME):
    print(f"\nERROR: File '{FILENAME}' not found!")
    print("\nOptions:")
    print("1. Set USE_KAGGLEHUB = True to automatically download the dataset")
    print("2. Download manually from: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")
    print("3. Update LOCAL_FILENAME variable with your file path")
    exit(1)

print(f"Reading CSV file: {FILENAME}")
df_mnist = pd.read_csv(FILENAME)

print(f"CSV shape: {df_mnist.shape}")
print(f"Columns: {df_mnist.columns[:5].tolist()}... (showing first 5)")

labels = df_mnist.iloc[:, 0].values
pixels = df_mnist.iloc[:, 1:].values

print(f"Labels shape: {labels.shape}")
print(f"Pixels shape: {pixels.shape}")

if pixels.shape[1] != 784:
    raise ValueError(f"Expected 784 pixel columns, but got {pixels.shape[1]}")

pixels = pixels / 255.0
num_samples = pixels.shape[0]
images = pixels.reshape(num_samples, 1, 28, 28)

images = images[:NUM_TRAIN_SAMPLES]
labels = labels[:NUM_TRAIN_SAMPLES]
num_samples = len(images)

print(f"Loaded {num_samples} images")
print(f"Image shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}\n")

conv1 = Conv2D(num_filters=8, in_channel=1, kernel_size=3)
activation_conv1 = Activation_LeakyReLU(alpha=0.01)
pool1 = MaxPool2D(pool_size=2, stride=2)

conv2 = Conv2D(num_filters=16, in_channel=8, kernel_size=3)
activation_conv2 = Activation_LeakyReLU(alpha=0.01)
pool2 = MaxPool2D(pool_size=2, stride=2)

flatten_layer = Flatten()

dense1_cnn = Layer_Dense(400, 128, weights_lambda_l2=lambda_l2_weights)
activation_dense1 = Activation_LeakyReLU(alpha=0.01)
dropout1_cnn = Layer_Dropout(dropout_rate)

dense2_cnn = Layer_Dense(128, 10, weights_lambda_l2=lambda_l2_weights)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
trainable_layers = [conv1, conv2, dense1_cnn, dense2_cnn]

num_batches = num_samples // batch_size

print("Starting CNN training with Numba JIT acceleration...")
print("First epoch will be slower (JIT compilation)...\n")

for epoch in range(epochs):
    epoch_loss = 0
    epoch_acc = 0

    indices = np.random.permutation(num_samples)
    images_shuffled = images[indices]
    labels_shuffled = labels[indices]

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size
        batch_images = images_shuffled[start_idx:end_idx]
        batch_labels = labels_shuffled[start_idx:end_idx]

        conv1.forward(batch_images)
        activation_conv1.forward(conv1.output)
        pool1.forward(activation_conv1.output)

        conv2.forward(pool1.output)
        activation_conv2.forward(conv2.output)
        pool2.forward(activation_conv2.output)

        flat_output = flatten_layer.forward(pool2.output)

        dense1_cnn.forward(flat_output)
        activation_dense1.forward(dense1_cnn.output)
        dropout1_cnn.forward(activation_dense1.output)

        dense2_cnn.forward(dropout1_cnn.output)

        loss = loss_activation.forward(dense2_cnn.output, batch_labels)

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == batch_labels)

        epoch_loss += np.mean(loss)
        epoch_acc += accuracy

        loss_activation.backward(loss_activation.output, batch_labels)
        dense2_cnn.backward(loss_activation.dinputs)

        dropout1_cnn.backward(dense2_cnn.dinputs)
        activation_dense1.backward(dropout1_cnn.dinputs)
        dense1_cnn.backward(activation_dense1.dinputs)

        flat_dinputs = flatten_layer.backward(dense1_cnn.dinputs)

        pool2.backward(flat_dinputs)
        activation_conv2.backward(pool2.dinputs)
        conv2.backward(activation_conv2.dinputs)

        pool1.backward(conv2.dinputs)
        activation_conv1.backward(pool1.dinputs)
        conv1.backward(activation_conv1.dinputs)

        optimizer.pre_update_params()
        for layer in trainable_layers:
            optimizer.update_params(layer)
        optimizer.post_update_params()

    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_acc / num_batches

    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f'Epoch: {epoch}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')

print("\nTraining completed!")

dropout1_cnn.set_prediction_mode()

print("\n" + "=")
print("MNIST DIGIT PREDICTION - Enter image data")
print("=")

while True:
    print(f"\nEnter row number from CSV (0-{num_samples - 1}) or 'quit' to exit:")
    user_input = input("Row number: ")

    if user_input.lower() == 'quit':
        break

    try:
        row_idx = int(user_input)

        if row_idx < 0 or row_idx >= num_samples:
            print(f"Invalid row number! Please enter a value between 0 and {num_samples - 1}")
            continue

        test_image = images[row_idx:row_idx + 1]
        true_label = labels[row_idx]

        conv1.forward(test_image)
        activation_conv1.forward(conv1.output)
        pool1.forward(activation_conv1.output)

        conv2.forward(pool1.output)
        activation_conv2.forward(conv2.output)
        pool2.forward(activation_conv2.output)

        flat_output = flatten_layer.forward(pool2.output)

        dense1_cnn.forward(flat_output)
        activation_dense1.forward(dense1_cnn.output)
        dropout1_cnn.forward(activation_dense1.output)

        dense2_cnn.forward(dropout1_cnn.output)
        loss_activation.activation.forward(dense2_cnn.output)

        prediction_probs = loss_activation.activation.output[0]
        predicted_digit = np.argmax(prediction_probs)
        confidence = prediction_probs[predicted_digit]

        print("\n" + "-")
        print(f"Row Index: {row_idx}")
        print(f"True Label: {true_label}")
        print(f"Predicted Digit: {predicted_digit}")
        print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")
        print(f"Correct: {'OK' if predicted_digit == true_label else '✗ NO'}")
        print(f"\nAll Class Probabilities:")
        for digit in range(10):
            bar = '█' * int(prediction_probs[digit] * 50)
            print(f"  Digit {digit}: {prediction_probs[digit]:.4f} {bar}")
        print("-")

        image_2d = test_image.reshape(28, 28)
        plt.figure(figsize=(4, 4))
        plt.imshow(image_2d, cmap='gray')
        plt.title(f'True: {true_label} | Predicted: {predicted_digit}')
        plt.axis('off')
        plt.show()

    except ValueError:
        print("Invalid input! Please enter a valid row number.")
    except Exception as e:
        print(f"Error: {e}")

print("\nExiting prediction mode. Goodbye!")