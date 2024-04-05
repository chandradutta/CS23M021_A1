from keras.datasets import fashion_mnist
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import argparse
#!pip install wandb
import wandb
from wandb.keras import WandbCallback
import socket
socket.setdefaulttimeout(30)
wandb.login()
wandb.init(project ='DL_Assignment_1')

# wandb.init(project="fashion-mnist-sample-images")

# Load the Fashion-MNIST dataset
# (x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()
# (x_train, y_train), (x_test,y_test) = mnist.load_data()

# print (len(train_images[0]))


# Defining class names
def labels():
  (x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  plt.figure(figsize=(10, 10))
  for i in range(len(class_names)):
      idx = next(idx for idx, label in enumerate(y_train) if label == i)
      plt.subplot(5, 5, i + 1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(x_train[idx], cmap=plt.cm.binary)
      plt.xlabel(class_names[i])
    wandb.log({f"example_{class_names[i]}": [wandb.Image(x_train[idx], caption=class_names[i])]})

# wandb.finish()



def softmax(ip):
    # ip=np.clip(ip,-500,500)
    eps = 1e-6
    return (np.exp(ip-max(ip)) / (sum(np.exp(ip-max(ip))) + eps))
def sigmoid(z):
    clipped_z=np.clip(z,-500, 500)
    return 1 / (1 + np.exp(-clipped_z))
def tanh(z):
  clipped_z = np.clip(z, -50, 50)
  return np.tanh(clipped_z)
def relu(Z):
  A = np.maximum(0,Z)
  return A
#derivatives
def tanh_derivative(z):
  return 1-np.tanh(z)**2
def relu_derivative(z):
  return np.where(z>0,1,0)
def sigmoid_derivative(z):
  return sigmoid(z) * (1 - sigmoid(z))


#weights initialization
def init(fun,hidLay,neurons):
  if(fun=='xavier'):
    return xavier_weights(hidLay,neurons)
  else:
    return weights(hidLay,neurons)

def weights(num_of_layers,n):
  ws=[]
  ns=[]
  bs=[]
  ws.append(np.random.randn(n,784))
  bs.append(np.random.randn(n,1))
  for i in range(1,num_of_layers):
    ws.append(np.random.randn(n,n))
    bs.append(np.random.randn(n,1))
  ws.append(np.random.randn(10,n))
  bs.append(np.random.randn(10,1))
  return ws,bs

def xavier_weights(num_of_layers,n):
  ws=[]
  ns=[]
  bs=[]
  ws.append(np.random.randn(n,784))
  bs.append(np.zeros((n,1)))
  for i in range(1,num_of_layers):
    ws.append(np.random.randn(n,n))
    bs.append(np.zeros((n,1)))
  ws.append(np.random.randn(10,n))
  bs.append(np.zeros((10,1)))
  return ws,bs






#forward

def calculate_pre_activation(weights, input_data, biases):
    return np.matmul(weights, input_data) + biases

def apply_activation(pre_activation, activation_function):
    if activation_function == 'sigmoid':
        return sigmoid(pre_activation)
    elif activation_function == 'tanh':
        return tanh(pre_activation)
    else :
        return relu(pre_activation)
def forward(x_train, ws, bs, activationfun, hidLay):
    x = x_train.reshape(784, 1) / 255.0
    pre_act = [0 for i in range(hidLay + 1)]
    activ = [0 for i in range(hidLay + 1)]

    for i in range(hidLay):
        if i == 0:
            pre_act[i] = calculate_pre_activation(ws[i], x, bs[i])
        else:
            pre_act[i] = calculate_pre_activation(ws[i], activ[i - 1], bs[i])

        activ[i] = apply_activation(pre_act[i], activationfun)
        # print(pre_act[i].shape)
        # print(activ[i].shape)
    pre_act[hidLay] = calculate_pre_activation(ws[hidLay], activ[hidLay - 1], bs[hidLay])
    activ[hidLay] = softmax(pre_act[hidLay])
    # print(pre_act[-1].shape)
    # print(activ[-1].shape)
    # prin(1)
    return activ, pre_act

#calculate back propagation

def calculate_gradients(d_a, activation_prev, theta, activationfun):
    d_w = np.matmul(d_a, activation_prev.T)
    d_b = np.copy(d_a)
    return d_w, d_b

def update_d_a(d_a, pre_activation_prev, theta, activationfun):
    d_h_prev = np.matmul(theta.T, d_a)

    if activationfun == 'sigmoid':
        d_a_new = np.multiply(d_h_prev, sigmoid_derivative(pre_activation_prev))
    elif activationfun == 'tanh':
        d_a_new = np.multiply(d_h_prev, tanh_derivative(pre_activation_prev))
    else:
        d_a_new = np.multiply(d_h_prev, relu_derivative(pre_activation_prev))

    return d_a_new



def compute(yt):
  e_l = np.zeros((10, 1))
  e_l[yt] = 1
  return e_l
def backwardPropagation(theta_w, activations, pre_activations, yt, hidLay, x, activationfun, l_fun):
    d_w = [0 for _ in range(hidLay + 1)]
    d_b = [0 for _ in range(hidLay + 1)]
    o_hot=compute(yt)
    if l_fun == 'entropy':
        d_a = -(o_hot - activations[hidLay])
    else:
        d_a = (activations[hidLay] - o_hot) * activations[hidLay] * (1 - activations[hidLay])

    layers = len(theta_w) - 1

    # while layers > 0:
    for layers in range(layers,0,-1):
        d_w[layers], d_b[layers] = calculate_gradients(d_a, activations[layers - 1], theta_w[layers], activationfun)
        d_a = update_d_a(d_a, pre_activations[layers - 1], theta_w[layers], activationfun)
        # layers -= 1

    d_w[0] = np.matmul(d_a, x.T)
    d_b[0] = np.copy(d_a)

    return d_w, d_b




def accuracy(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun):
  acc=0
  val=0
  for i in range(54000):
    act,p_act=forward(x_train[i],theta_w,theta_b,activationfun,hidLay)
    # print(np.argmax(act[-1]),end=",")
    if(np.argmax(act[-1])==y_train[i]):
     acc=acc+1
    temp=act[-1]
    if(l_fun=="entropy"):
     val=val-np.log(temp[y_train[i]]+(1e-5))
    elif (l_fun=="mean_squared_error"):
     val+=np.sum((y_train[i] -act[-1]) ** 2)
  return (acc/540),val
  # return (acc/540)

def validation(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun):
  accuracy=0
  val=0
  start_index = int(len(x_train) * 0.1)

  for i in range(start_index, len(x_train)):
    act,p_act=forward(x_train[i],theta_w,theta_b,activationfun,hidLay)
    if(np.argmax(act[-1])==y_train[i]):
      accuracy+=1
    temp=act[-1]
    val=val-np.log(temp[y_train[i]]+(1e-5))
  return (accuracy/(int(len(x_train))))*100,val

def accuracy_conf(theta_w,theta_b,x_test,y_test,activationfun,hidLay,l_fun,found,real):
  acc=0
  val=0
  for i in range(10000):
    act,p_act=forward(x_test[i],theta_w,theta_b,activationfun,hidLay)
    # print(np.argmax(act[-1]),end=",")
    found.append(np.argmax(act[-1]))
    real.append(y_test[i])
  return found,real





def do_sgd(theta_w,theta_b,x_train,y_train, learning_rate, max_iterations,activationfun,hidLay,l_fun):
  for ii in range(max_iterations):

    for xt,yt in zip(x_train,y_train):
      # print(xt.shape)
      activations, pre_activations= forward(xt,theta_w,theta_b,activationfun,hidLay)
      # activations.reverse()
      # print(activations[-1].shape)
      x = xt.reshape(784,1) / 255.0
      gradients_weights, gradients_biases = backwardPropagation(theta_w,activations,pre_activations,yt,hidLay,x,activationfun,l_fun)
      for i in range(len(theta_w)):
        theta_w[i] = theta_w[i] - learning_rate * gradients_weights[i]
        theta_b[i] = theta_b[i] - learning_rate * gradients_biases[i]
    acc,loss=accuracy(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun)
    v_acc,v_loss=validation(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun)
    print(acc,loss)
    print(v_acc,v_loss)
    wandb.log({"Train_Accuracy" : acc})
    wandb.log({"Train_Loss" : loss})
    wandb.log({"Validation_acc" : v_acc})
    wandb.log({"Validation_loss" : v_loss})
    wandb.log({"epoch" : max_iterations})




def do_mgd(max_epochs, x_train, y_train, theta_w, theta_b, eta, beta,weight_decay,activationfun,batch_size,l_fun,hidLay):
    prev_dw = [np.zeros_like(w) for w in theta_w]  # Initialize previous gradients for weights
    prev_db = [np.zeros_like(w) for w in theta_b]  # Initialize previous gradients for biases

    for epoch in range(max_epochs):
        t=1
        d_w = [np.zeros_like(w) for w in theta_w]
        d_b = [np.zeros_like(w) for w in theta_b]
        # print(d_w[0].shape,theta_w[0].shape)
        for xt,yt in zip(x_train,y_train):
            # print(xt.shape)
            activations, pre_activations= forward(xt,theta_w,theta_b,activationfun,hidLay)
            # activations.reverse()
            # print(activations[-1].shape)
            x = xt.reshape(784,1) / 255.0
            gradients_weights, gradients_biases = backwardPropagation(theta_w,activations,pre_activations,yt,hidLay,x,activationfun,l_fun)
            # Compute squared gradients and update parameters
            for i in range(len(theta_w)):
                d_w[i] += gradients_weights[i]
                d_b[i] += gradients_biases[i]
            if((t%batch_size)==0):
              for i in range(len(theta_w)):
                prev_dw[i] = beta*prev_dw[i] + d_w[i]
                prev_db[i] = beta*prev_db[i] + d_b[i]

                theta_b[i] = theta_b[i] - eta*prev_db[i]
                theta_w[i] = theta_w[i] - eta*prev_dw[i]-weight_decay*theta_w[i]

              d_w = [np.zeros_like(w) for w in theta_w]
              d_b = [np.zeros_like(w) for w in theta_b]

            t=t+1
        acc,loss=accuracy(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun)
        v_acc,v_loss=validation(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun)
        print(acc,loss)
        print(v_acc,v_loss)
        wandb.log({"Train_Accuracy" : acc})
        wandb.log({"Train_Loss" : loss})
        wandb.log({"Validation_acc" : v_acc})
        wandb.log({"Validation_loss" : v_loss})
        wandb.log({"epoch" : epoch})





def do_nag(max_epochs, x_train, y_train, theta_w, theta_b, eta, beta,weight_decay,activationfun,batch_size,l_fun,hidLay):
    prev_dw = [np.zeros_like(w) for w in theta_w]  # Initialize previous gradients for weights
    prev_db = [np.zeros_like(b) for b in theta_b]  # Initialize previous gradients for biases

    for epoch in range(max_epochs):
        t=1
        d_w = [np.zeros_like(w) for w in theta_w]
        d_b = [np.zeros_like(b) for b in theta_b]
        d_wx = [np.zeros_like(w) for w in theta_w]
        d_bx = [np.zeros_like(b) for b in theta_b]
        for xt,yt in zip(x_train,y_train):

            lookahead_weights = [w - beta * prev_dw[i] for i, w in enumerate(theta_w)]
            lookahead_biases = [b - beta * prev_db[i] for i, b in enumerate(theta_b)]
            activations, pre_activations= forward(xt,lookahead_weights,lookahead_biases,activationfun,hidLay)
            # activations.reverse()
            # print(activations[-1].shape)
            x = xt.reshape(784,1) / 255.0
            gradients_weights, gradients_biases = backwardPropagation(theta_w,activations,pre_activations,yt,hidLay,x,activationfun,l_fun)

            for i in range(len(theta_w)):
              d_w[i] += gradients_weights[i]
              d_b[i] += gradients_biases[i]
            if(t)%batch_size==0:
              for i in range(len(theta_w)):
                prev_dw[i] = beta*prev_dw[i] + d_w[i]
                prev_db[i] = beta*prev_db[i] + d_b[i]

                theta_b[i] = theta_b[i] - eta*prev_db[i]
                theta_w[i] = theta_w[i] - eta*prev_dw[i]
                # d_wx[i] = beta * prev_dw[i] + eta * d_w[i]  # Compute momentum-based gradient for weights
                # d_bx[i] = beta * prev_db[i] + eta * d_b[i]  # Compute momentum-based gradient for biases
                #   # Update weights and biases
                # theta_w[i] -= (d_wx[i])-weight_decay*theta_w[i]
                # theta_b[i] -= d_bx[i]

                # # Update previous gradients for the next iteration
                # prev_dw[i] = d_wx[i]
                # prev_db[i] = d_bx[i]
              d_w = [np.zeros_like(w) for w in theta_w]
              d_b = [np.zeros_like(w) for w in theta_b]
            t+=1
        acc,loss=accuracy(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun)
        v_acc,v_loss=validation(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun)
        print(acc,loss)
        print(v_acc,v_loss)
        wandb.log({"Train_Accuracy" : acc})
        wandb.log({"Train_Loss" : loss})
        wandb.log({"Validation_acc" : v_acc})
        wandb.log({"Validation_loss" : v_loss})
        wandb.log({"epoch" : epoch})





def do_rmsprop(max_epochs, x_train, y_train, theta_w, theta_b, learning_rate, beta,weight_decay,activationfun,batch_size,l_fun,hidLay):
    # Initialization
    v_dw = [np.zeros_like(w) for w in theta_w]  # Initialize squared gradients for weights
    v_db = [np.zeros_like(b) for b in theta_b]  # Initialize squared gradients for biases
    eps = 1e-4  # Small constant to prevent division by zero

    for epoch in range(max_epochs):
        d_w = [np.zeros_like(w) for w in theta_w]
        d_b = [np.zeros_like(b) for b in theta_b]
        t=1
        for xt,yt in zip(x_train,y_train):
            activations, pre_activations= forward(xt,theta_w,theta_b,activationfun,hidLay)
            # Backward propagation
            x = xt.flatten().reshape(784,1)

            gradients_weights, gradients_biases =backwardPropagation(theta_w,activations,pre_activations,yt,hidLay,x,activationfun,l_fun)
            # Compute squared gradients and update parameters

            for i in range(len(theta_w)):
                d_w[i] += gradients_weights[i]
                d_b[i] += gradients_biases[i]

                # v_dw[i] = beta * v_dw[i] + (1 - beta) * gradients_weights[i] ** 2  # RMSprop update for weights
                # v_db[i] = beta * v_db[i] + (1 - beta) * gradients_biases[i] ** 2  # RMSprop update for biases


            if(t%batch_size==0):
              for k in range(len(theta_w)):
                v_dw[k] = (1 - beta) * (d_w[k] ** 2) + beta * v_dw[k]  # RMSprop update for weights
                v_db[k] = (1 - beta) * (d_b[k] ** 2) + beta * v_db[k]
                # Update weights and biases
                theta_w[k] -= (learning_rate*d_w[k])/ (np.sqrt(v_dw[k]) + eps)-weight_decay*theta_w[k]
                theta_b[k] -= (learning_rate*d_b[k])/ (np.sqrt(v_db[k])+ eps)
            t+=1

        acc,loss=accuracy(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun)
        v_acc,v_loss=validation(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun)
        print(acc,loss)
        print(v_acc,v_loss)
        wandb.log({"Train_Accuracy" : acc})
        wandb.log({"Train_Loss" : loss})
        wandb.log({"Validation_acc" : v_acc})
        wandb.log({"Validation_loss" : v_loss})
        wandb.log({"epoch" : epoch})





def do_adam(max_epochs, x_train, y_train, theta_w, theta_b, eta, beta1, beta2,weight_decay,activationfun,batch_size,hidLay,l_fun):
    # Initialization
    m_dw = [np.zeros_like(w) for w in theta_w]  # Initialize first moment for weights
    m_db = [np.zeros_like(w) for w in theta_b]  # Initialize first moment for biases
    v_dw = [np.zeros_like(w) for w in theta_w]  # Initialize second moment for weights
    v_db = [np.zeros_like(w) for w in theta_b]  # Initialize second moment for biases
    eps = 1e-8  # Small constant to prevent division by zero
    # t = 0  # Time step initialization

    for epoch in range(max_epochs):
        d_w = [np.zeros_like(w) for w in theta_w]
        d_b = [np.zeros_like(b) for b in theta_b]
        t=1
        for xt,yt in zip(x_train,y_train):
            activations, pre_activations= forward(xt,theta_w,theta_b,activationfun,hidLay)
            # Backward propagation
            x = xt.flatten().reshape(784,1)/255.0

            gradients_weights, gradients_biases =backwardPropagation(theta_w,activations,pre_activations,yt,hidLay,x,activationfun,l_fun)
            for i in range(len(theta_w)):
                d_w[i] += gradients_weights[i]
                d_b[i] += gradients_biases[i]
            if(t)%batch_size==0:

              # Update biased first moment estimates
              for i in range(len(theta_w)):
                  # print(len(theta_w))
                  m_dw[i] =  (1 - beta1) * d_w[i] +beta1 * m_dw[i] # Update first moment for weights
                  m_db[i] =  (1 - beta1) * d_b[i]+beta1 * m_db[i]   # Update first moment for biases
                  v_dw[i] =  (1 - beta2) * (d_w[i] ** 2)  +beta2 * v_dw[i]  # Update second moment for weights
                  v_db[i] =  (1 - beta2) * (d_b[i] ** 2) +beta2 * v_db[i]  # Update second moment for biases

                  # Correct bias in first moment
                  m_dw_corrected = m_dw[i] / (1 - beta1 ** epoch+1)  # Correct first moment for weights
                  m_db_corrected = m_db[i] / (1 - beta1 ** epoch+1)  # Correct first moment for biases

                  v_dw_corrected = v_dw[i] / (1 - beta2 ** epoch+1) # Correct second moment for weights
                  v_db_corrected = v_db[i] / (1 - beta2 ** epoch+1)  # Correct second moment for biases

                  # Update parameters
                  theta_w[i] -= (eta * m_dw_corrected) / (np.sqrt(v_dw_corrected) + eps)-(weight_decay*theta_w[i]) # Update weights
                  theta_b[i] -= (eta * m_db_corrected)/ (np.sqrt(v_db_corrected) + eps)  # Update biases
              d_w = [np.zeros_like(w) for w in theta_w]
              d_b = [np.zeros_like(w) for w in theta_b]
            t+=1
        acc,loss=accuracy(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun)
        v_acc,v_loss=validation(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun)
        print(acc,loss)
        print(v_acc,v_loss)
        wandb.log({"Train_Accuracy" : acc})
        wandb.log({"Train_Loss" : loss})
        wandb.log({"Validation_acc" : v_acc})
        wandb.log({"Validation_loss" : v_loss})
        wandb.log({"epoch" : epoch})




def do_nadam(max_epochs, x_train, y_train, theta_w, theta_b, eta, beta1, beta2,weight_decay,activationfun,batch_size,hidLay,l_fun):
    # Initialization
    m_dw = [np.zeros_like(w) for w in theta_w]  # Initialize first moment for weights
    m_db = [np.zeros_like(w) for w in theta_b]  # Initialize first moment for biases
    v_dw = [np.zeros_like(w) for w in theta_w]  # Initialize second moment for weights
    v_db = [np.zeros_like(w) for w in theta_b]  # Initialize second moment for biases
    eps = 1e-8  # Small constant to prevent division by zero
    # t = 0  # Time step initialization

    for epoch in range(max_epochs):
        d_w = [np.zeros_like(w) for w in theta_w]
        d_b = [np.zeros_like(b) for b in theta_b]
        t=1
        for xt,yt in zip(x_train,y_train):
            activations, pre_activations= forward(xt,theta_w,theta_b,activationfun,hidLay)
            # Backward propagation
            x = xt.flatten().reshape(784,1)/255.0

            gradients_weights, gradients_biases =backwardPropagation(theta_w,activations,pre_activations,yt,hidLay,x,activationfun,l_fun)
            for i in range(len(theta_w)):
                d_w[i] += gradients_weights[i]
                d_b[i] += gradients_biases[i]
            if(t)%batch_size==0:

              # Update biased first moment estimates
              for i in range(len(theta_w)):
                  # print(len(theta_w))
                  m_dw[i] =  (1 - beta1) * d_w[i] +beta1 * m_dw[i] # Update first moment for weights
                  m_db[i] =  (1 - beta1) * d_b[i]+beta1 * m_db[i]   # Update first moment for biases
                  v_dw[i] =  (1 - beta2) * d_w[i] ** 2  +beta2 * v_dw[i]  # Update second moment for weights
                  v_db[i] =  (1 - beta2) * d_b[i] ** 2 +beta2 * v_db[i]  # Update second moment for biases


                  m_dw_corrected = m_dw[i] / (1 - beta1 ** epoch+1)  # Correct first moment for weights
                  m_db_corrected = m_db[i] / (1 - beta1 ** epoch+1)  # Correct first moment for biases

                  v_dw_corrected = v_dw[i] / (1 - beta2 ** epoch+1) # Correct second moment for weights
                  v_db_corrected = v_db[i] / (1 - beta2 ** epoch+1)  # Correct second moment for biases

                  theta_w[i] = theta_w[i] - (eta / (np.sqrt(v_dw_corrected+eps)))*(beta1 * m_dw_corrected + (1-beta1)*d_w[i]/(1-beta1**(epoch+1))) - (weight_decay * theta_w[i])
                  theta_b[i] = theta_b[i] - (eta / (np.sqrt(v_db_corrected+eps)))*(beta1 * m_db_corrected + (1-beta1)*d_b[i]/(1-beta1**(epoch+1)))
              d_w = [np.zeros_like(w) for w in theta_w]
              d_b = [np.zeros_like(w) for w in theta_b]
            t+=1
        acc,loss=accuracy(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun)
        v_acc,v_loss=validation(theta_w,theta_b,x_train,y_train,activationfun,hidLay,l_fun)
        print(acc,loss)
        print(v_acc,v_loss)
        wandb.log({"Train_Accuracy" : acc})
        wandb.log({"Train_Loss" : loss})
        wandb.log({"Validation_acc" : v_acc})
        wandb.log({"Validation_loss" : v_loss})
        wandb.log({"epoch" : epoch})
    # return theta_w,theta_b





def evaluate_model(max_epochs, hidLay, n, eta, beta1, beta2, batch_size, activationfun, weight_decay, l_fun, theta_w, theta_b):
      # Train the model
      found = []
      real = []
      w, b = do_nadam(max_epochs, x_train, y_train, theta_w, theta_b, eta, beta1, beta2, weight_decay, activationfun, batch_size, hidLay, l_fun)

      # Evaluate the model
      u, v = accuracy_conf(w, b, x_test, y_test, activationfun, hidLay, l_fun, found, real)

      # Generate confusion matrix
      conf = confusion_matrix(real, found)

      # Plot confusion matrix
      plt.figure(figsize=(10, 10))
      sn.heatmap(conf, annot=True, fmt='d', cmap='BuPu', linewidths=2, cbar=True, linecolor='black',
                xticklabels=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
                yticklabels=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
      plt.xlabel("LIKELY")
      plt.ylabel("REAL")
      plt.title('Confusion Matrix')
      plt.savefig('conf_matrix_1.png')

      # Log confusion matrix to wandb
      wandb.log({'conf_matrix': wandb.Image('conf_matrix_1.png')})

      # Show plot
      plt.show()

      
def main_function(optimization_algorithm,max_epochs, x_train, y_train, theta_w, theta_b,learning_rate, beta,beta1,weight_decay,activationfun,batch_size,l_fun,hidLay,b,momentum,epsilon):
  if optimization_algorithm == 'momentum':
    do_mgd(max_epochs, x_train, y_train, theta_w, theta_b, learning_rate, beta,weight_decay,activationfun,batch_size,l_fun,hidLay)
  elif optimization_algorithm == 'ngd':
    do_nag(max_epochs, x_train, y_train, theta_w, theta_b, learning_rate, beta,weight_decay,activationfun,batch_size,l_fun,hidLay)
  elif optimization_algorithm == 'rmsprop':
      do_rmsprop(max_epochs, x_train, y_train, theta_w, theta_b, learning_rate, beta,weight_decay,activationfun,batch_size,l_fun,hidLay)
  elif optimization_algorithm == 'adam':
    do_adam(max_epochs, x_train, y_train, theta_w, theta_b, learning_rate, beta, beta1,weight_decay,activationfun,batch_size,hidLay,l_fun)
  elif optimization_algorithm == 'nadam':
    do_nadam(max_epochs, x_train, y_train, theta_w, theta_b, learning_rate, beta, beta1,weight_decay,activationfun,batch_size,hidLay,l_fun)
  else:
    do_sgd(theta_w,theta_b,x_train,y_train, learning_rate, max_epochs,activationfun,hidLay,l_fun)


def parse_arguments():
  parser = argparse.ArgumentParser(description='Training Parameters')
  parser.add_argument('-wp', '--wandb_project', type=str, default='DL_Assignment_1',
                        help='Project name')
  
  parser.add_argument('-we', '--wandb_entity', type=str, default='Entity_DL',
                        help='Wandb Entity')
  
  parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',choices=["mnist", "fashion_mnist"],
                        help='Dataset choice: fashion_mnist , mnist')
  
  parser.add_argument('-e', '--epochs', type=int, default=10,help='Number of epochs for training network')

  parser.add_argument('-b', '--batch_size', type=int, default=32,help='Batch size for training neural network')

  parser.add_argument('-l', '--loss', type=str, default='entropy',choices=["cross_entropy", "mean_squared_error"],help='Choice of mean_squared_error or cross_entropy')
  
  parser.add_argument('-o', '--optimizer', type=str, default='nadam', choices = ["sgd", "momentum", "ngd", "rmsprop", "adam", "nadam"],help='Choice of optimizer')
   
  parser.add_argument('-lr', '--learning_rate', type=int, default=0, help='Learning rate')

  parser.add_argument( '-m', '--momentum', type=int, default=0.5, help='Momentum parameter')

  parser.add_argument('-beta', '--beta', type=int, default=0.5, help='Beta parameter')

  parser.add_argument('-beta1', '--beta1', type=int, default=0.9, help='Beta1 parameter')

  parser.add_argument('-beta2', '--beta2', type=int, default=0.999, help='Beta2 parameter')

  parser.add_argument( '-eps', '--epsilon', type=int, default=0.000001, help='Epsilon used by optimizers')

  parser.add_argument( '-w_i', '--weight_init',type=str, default="xavier",choices=["random", "xavier"], help='randomizer for weights')

  parser.add_argument('-w_d','--weight_decay',  type=int, default=0, help='Weight decay parameter')

  parser.add_argument( '-nhl', '--num_layers',type=int, default=3, help='Number of hidden layers')
  
  parser.add_argument( '-sz','--hidden_size', type=int, default=128, help='Number of neurons in each layer')

  parser.add_argument( '-a','--activation', type=str, default="sigmoid",choices=["sigmoid", "tanh", "ReLU"], help='activation functions')

  return parser.parse_args()

args = parse_arguments()



wandb.init(project=args.wandb_project)

if args.dataset == 'mnist':
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
   
else:
    (x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

wandb.run.name=f'activation {args.activation} weight_init{args.weight_init}opt{args.optimizer}'
theta_w,theta_b=init(args.weight_init,args.num_layers,args.hidden_size)
main_function(args.optimizer,args.epochs,x_train, y_train,theta_w, theta_b,args.learning_rate,args.beta1, args.beta2,args.weight_decay,args.activation,args.batch_size,args.loss,args.num_layers, args.beta, args.momentum,args.epsilon)
