CNN from Scratch in NumPy

This project is a complete Convolutional Neural Network (CNN) built entirely from scratch using only NumPy. It is written to demonstrate a deep, fundamental understanding of all the core components of a modern deep learning framework, including the complex mathematics of the forward and backward passes.

This network is trained on the MNIST dataset of handwritten digits, which it downloads automatically using the Kaggle API.

Core Features Implemented

This project is not a simple script; it's a complete object-oriented framework that includes:

100% NumPy: No TensorFlow, Keras, or PyTorch. All calculations are built from the ground up.

Class-Based Layers: Every component is a modular Python class:

Conv2D (with 2 hidden layers)

MaxPool2D

Flatten

Layer_Dense

Layer_Dropout

Full Backpropagation: A complete .backward() method for every layer, including the complex gradient routing for MaxPool2D and the convolutional gradients (dweights, dbiases, dinputs) for Conv2D.

Advanced Optimizer: A full implementation of the Adam optimizer, including beta1 (momentum), beta2 (RMSprop), and bias correction (m_hat, v_hat).

Modern Activation: LeakyReLU for hidden layers and Softmax for the output.

Regularization: Includes both L2 Weight Decay (penalizes large weights) and Dropout (randomly deactivates neurons during training) to prevent overfitting.

Stable Loss Function: Uses a combined Activation_Softmax_Loss_CategoricalCrossentropy class for an efficient and numerically stable backward pass.

Mini-Batch Training: Loads the full dataset and trains using mini-batches with shuffling.

Input Scaling: Normalizes pixel data (/ 255.0) for stable training.

Interactive Prediction: After training, the script enters a loop allowing you to test any image from the test set by its row number and see the prediction visualize with matplotlib.

 Network Architecture

The model is a 2-layer CNN with the following architecture:

+---------------------+   Input: (N, 1, 28, 28)
|    Input Image      |
+---------------------+
           |
           v
+---------------------+   Conv2D (8 filters, 3x3)
| Conv1 (N, 8, 26, 26)  |   -> (N, 8, 26, 26)
+---------------------+
           |
           v
+---------------------+
|      LeakyReLU      |
+---------------------+
           |
           v
+---------------------+   MaxPool2D (2x2)
| Pool1 (N, 8, 13, 13)  |   -> (N, 8, 13, 13)
+---------------------+
           |
           v
+---------------------+   Conv2D (16 filters, 3x3)
| Conv2 (N, 16, 11, 11) |   -> (N, 16, 11, 11)
+---------------------+
           |
           v
+---------------------+
|      LeakyReLU      |
+---------------------+
           |
           v
+---------------------+   MaxPool2D (2x2)
| Pool2 (N, 16, 5, 5)   |   -> (N, 16, 5, 5)
+---------------------+
           |
           v
+---------------------+   Flatten
| Flatten (N, 400)    |   -> (N, 400)  (16 * 5 * 5)
+---------------------+
           |
           v
+---------------------+   Dense (400 -> 128)
| Dense1 (N, 128)     |
+---------------------+
           |
           v
+---------------------+
|      LeakyReLU      |
+---------------------+
           |
           v
+---------------------+   Dropout (Rate: 0.3)
|   Dropout (N, 128)  |
+---------------------+
           |
           v
+---------------------+   Dense (128 -> 10)
| Dense2 (N, 10)      |
+---------------------+
           |
           v
+---------------------+   Softmax
| Output Probs (N, 10)|
+---------------------+


⚙️ How to Run

Install Dependencies:
You will need numpy, pandas, matplotlib, and kagglehub.

pip install numpy pandas matplotlib kagglehub



Kaggle API Credentials (First Time Only):
kagglehub needs your Kaggle API credentials. The easiest way is to:

Go to your Kaggle account settings (https.kaggle.com/<your-username>/account).

Click "Create New API Token" to download kaggle.json.

Place this file in the correct location (e.g., C:\Users\<Your-Username>\.kaggle\kaggle.json on Windows).

Run the Script:

python your_script_name.py



The script will automatically download and cache the MNIST dataset (as mnist_test.csv) on its first run.

It will train the model for the number of epochs specified, printing the average loss and accuracy.

After training, it will enter an interactive prediction loop. Type in a row number (e.g., 122) to see the model predict that digit and show you the image.

Type quit to exit.
