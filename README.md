
**CS6910: Fundamentals of Deep Learning**
This repository contains implementations done as part of the learning in CS6910: Fundamentals of Deep Learning course offered by Indian Institute of Technology, Madras, taught by Prof. Mitesh Khapra.

**Overview**
In this repository, I have implemented a feedforward neural network and written the backpropagation code for training the network. All matrix/vector operations are performed using the NumPy library. The network is trained and tested using the Fashion-MNIST dataset, where the task involves classifying input images, each consisting of 784 pixels (28 x 28), into one of 10 classes.

**Major Implementations**
1. Feedforward Neural Network
     *calculate_pre_activation: Computes pre-activation values by multiplying weights with input data and adding biases.
     *apply_activation: Applies the specified activation function to the pre-activation values.
     *forward:
          Reshapes and normalizes input data.
          Computes pre-activation and activation values for each hidden layer.
          Applies softmax activation to the output layer.
          Returns activation and pre-activation values for all layers.
2. Backpropagation Algorithm
     *Error Minimization: Backpropagation minimizes the error between the network's output and the expected output.
     *Gradient Descent: Utilizes gradient descent to adjust weights and biases for error reduction.
     *Backward Pass: Computes gradients backward through the network using the chain rule.
     *Weight Update: Updates weights and biases based on computed gradients to reduce error.
     *Learning Rate: Controls the size of weight updates during gradient descent.
     *Iterative: Iteratively adjusts weights and biases until convergence to a satisfactory solution.
     *Local Minima: May converge to local minima, addressed with techniques like momentum and adaptive learning rates.
3. Gradient Descent Variants
   a) Stochastic Gradient Descent
   b) Momentum Based Gradient Descent
   c) Nesterov Accelerated Gradient Descent
   d) RMSProp
   e) Adam
   f) Nadam

**Training the Model**
     The model is trained by calling the main function, which is integrated with WandB for experiment tracking. Various parameters are tuned during training:
    * Number of epochs: 5, 10
    * Number of hidden layers: 3, 4, 5
    * Size of every hidden layer: 32, 64, 128
    * Weight decay (L2 regularization): 0, 0.0005, 0.5
    * Learning rate: 1e-3, 1e-4
    * Optimizer: SGD, Momentum, Nesterov, RMSProp, Adam, Nadam
    * Batch size: 16, 32, 64
    * Weight initialization: Random, Xavier
    * Activation functions: Sigmoid, Tanh, ReLU

**Libraries**
  *Required libraries such as NumPy, WandB, scikit-learn, Matplotlib, and Seaborn are imported to support the implementation.

**Confusion Matrix**
  *The best optimizer is evaluated on the test data, and the results are visualized using a confusion matrix.
