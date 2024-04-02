Description:
This repository is done as part of the learning CS6910: Fundamentals of Deep Learning course offered by Indian Institute of Technology, Madras. The course is taught by Prof. Mitesh Khapra.
The following are the implementations that are done as a part of the exercise:
•	I have implemented a feedforward neural network and written the backpropagation code for training the network.
•	I utilized numpy for all matrix/vector operations.
•	The network has been trained and tested using the Fashion-MNIST dataset. Specifically, the task involved classifying input images, each consisting of 784 pixels (28 x 28), into one of 10 classes.
•	Before implementing the following functions the data is extracted and saved as train and test
Major Implementations:
1)Feedforward Neural Network:
•	calculate_pre_activation: Computes pre-activation values by multiplying weights with input data and adding biases.
•	apply_activation: Applies the specified activation function to the pre-activation values.
•	forward:
•	Reshapes and normalizes input data.
•	Computes pre-activation and activation values for each hidden layer.
•	Applies softmax activation to output layer.
•	Returns activation and pre-activation values for all layers.
2)Backpropagation Algorithm:
•	Error Minimization: Backpropagation minimizes the error between the network's output and the expected output.
•	Gradient Descent: Utilizes gradient descent to adjust weights and biases for error reduction.
•	Backward Pass: Computes gradients backward through the network using the chain rule.
•	Weight Update: Updates weights and biases based on computed gradients to reduce error.
•	Learning Rate: Controls the size of weight updates during gradient descent.
•	Iterative: Iteratively adjusts weights and biases until convergence to a satisfactory solution.
•	Local Minima: May converge to local minima, addressed with techniques like momentum and adaptive learning rates.
3)Gradient Descent's variants like:
     a) Stochastic Gradient Descent
     b) Momentum Based Gradient Descent
     c) Nesterov Accelerated Gradient Descent
     d) RMSProp
     e) Adam
     f) Nadam

Training the Model:
 The Model is trained by calling the main function which is integrated to wandb using the following parameters:
  •  number of epochs: 5, 10
  •  number of hidden layers: 3, 4, 5
  •  size of every hidden layer: 32, 64, 128
  •  weight decay (L2 regularisation): 0, 0.0005, 0.5
  •  learning rate: 1e-3, 1 e-4 
  •  optimizer: sgd, momentum, nesterov, rmsprop, adam, nadam
  •  batch size: 16, 32, 64
  •  weight initialisation: random, Xavier
  •  activation functions: sigmoid, tanh, ReLU

Libraries:
•	Required libraries numpy, wandb, scikit-learn, matplotlib, and seaborn are imported.
Confusion Matrix:
•	The best optimizer is tested on the test data and the results are plotted in a confusion matrix.
