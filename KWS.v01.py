import matplotlib
import numpy as np
import librosa  # for audio related library using

import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
font = {'weight': 'normal',
        'size': 22}
matplotlib.rc('font', **font)


# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
np.random.seed(102)
step_size = 0.001
batch_size = 10000
max_epochs = 1000

# GLOBAL PARAMETERS FOR NETWORK ARCHITECTURE
# input 1:    11x49x1 --> conv 4x10/2                   --> 6x25x64
# layer 2~5:  6x25x64 --> dw_conv 3x3/1, pw_conv 1x1/1  --> 6x25x64
# layer 6:   6x25x64 --> grand average                 --> 1x1x64
# output:     1x1x64  --> fully_connected               --> 1x1x10
number_of_hidden_layers = 4
number_of_hidden_lnodes = 64  # only for layer > 1
activation = "ReLU"


def main():

    # Load data and display an example
    X_train, Y_train, X_val, Y_val, X_test = loadData()
    # display input feature (X_train[np.random.randint(7,len(X_train))])

    # Build a network with input feature dimensions, output feature dimension,
    # hidden dimension, and number of layers as specified below

    # input parameter (#### ERROR found ####)
    kws_network = CLASS_KWS_NETWORK(
        X_train.shape[1], Y_train, number_of_hidden_layers, number_of_hidden_lnodes, activation=activation)

    # Some lists for book-keeping for plotting later
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    # Loss function
    lossFunc = CrossEntropySoftmax()

    # Indicies we will use to shuffle data randomly
    inds = np.arange(len(X_train))
    for i in range(max_epochs):
        # Shuffled indicies so we go through data in new random batches
        np.random.shuffle(inds)
        # Go through all datapoints once (aka an epoch)
        j = 0
        acc_running = loss_running = 0
        while j < len(X_train):
            # Select the members of this random batch
            b = min(batch_size, len(X_train)-j)
            X_batch = X_train[inds[j:j+b]]
            # change type in int
            Y_batch = Y_train[inds[j:j+b]].astype(np.int32)
            # Compute the scores for our 10 classes using our model
            logits = kws_network.forward(X_batch)
            loss = lossFunc.forward(logits, Y_batch)
            # accuracy
            acc = np.mean(np.argmax(logits, axis=1)[:, np.newaxis] == Y_batch)
            # Compute gradient of Cross-Entropy Loss with respect to logits
            loss_grad = lossFunc.backward()
            # Pass gradient back through networks
            kws_network.backward(loss_grad)
            # Take a step of gradient descent
            kws_network.step(step_size)
            # Record losses and accuracy then move to next batch
            train_losses.append(loss)
            train_accs.append(acc)
            loss_running += loss*b
            acc_running += acc*b
            j += batch_size
        # Evaluate performance on validation. This function looks very similar to the training loop above,
        vloss, vacc = evaluateValidation(kws_network, X_val, Y_val, batch_size)
        val_losses.append(vloss)
        val_accs.append(vacc)

        # Print out the average stats over this epoch
        logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(
            i, loss_running/len(X_train), acc_running / len(X_train)*100, vacc*100))

# -------------------------------------------------------------------------------------------------------------------------
#  fig, ax1 = plt.subplots(figsize=(16,9))
#  color = 'tab:red'
#  ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
#  ax1.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_losses))], val_losses,c="red", label="Val. Loss")
#  ax1.set_xlabel("Iterations")
#  ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
#  ax1.tick_params(axis='y', labelcolor=color)
#  ax1.set_ylim(-0.01,3)
#
#  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#  color = 'tab:blue'
#  ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
#  ax2.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_accs))], val_accs,c="blue", label="Val. Acc.")
#  ax2.set_ylabel(" Accuracy", c=color)
#  ax2.tick_params(axis='y', labelcolor=color)
#  ax2.set_ylim(-0.01,1.01)
#
#  fig.tight_layout()  # otherwise the right y-label is slightly clipped
#  ax1.legend(loc="center")
#  ax2.legend(loc="center right")
#  plt.show()
# -------------------------------------------------------------------------------------------------------------------------


class CLASS_CONV:
    def __init__(self, input_dim, output_dim):
        # WOON - need to make sure the rati of weight
        self.weights = np.random.randn(input_dim, output_dim)*4*10
        # Reduced size of byte since bias doesn't have to be large
        self.bias = np.ones((1, output_dim))*0.5

    # During the forward pass, we simply compute Xw+b
    def forward(self, input):
        self.input = input
        return self.input@self.weights + self.bias

    def backward(self, grad):
        #   dL/d_input = (dL/d_output) * (d_output/d_input)
        #                          where (d_output/d_input) = wT
        #
        grad_input = grad@np.transpose(self.weights)
        # compute gradient w.r.t. weights and biases
        # dL/dW = (input)T * (dL/d_Z)
        # dL/dB = Sum of (dL/dZ)
        self.grad_weights = np.transpose(self.input)@grad
        self.grad_bias = np.sum(grad, axis=0)
        return grad_input

    def step(self, step_size):
        self.weights -= step_size * self.grad_weights
        self.bias -= step_size * self.grad_bias


class CLASS_D_CONV:
    def __init__(self, input_dim, output_dim):
        # WOON - need to make sure the rati of weight
        self.weights = np.random.randn(input_dim, output_dim)*3*3
        self.bias = np.ones((1, output_dim))

    # During the forward pass, we simply compute Xw+b
    def forward(self, input):
        self.input = input
        return self.input@self.weights + self.bias

    def backward(self, grad):
        #   dL/d_input = (dL/d_output) * (d_output/d_input)
        #                          where (d_output/d_input) = wT
        #
        grad_input = grad@np.transpose(self.weights)
        # compute gradient w.r.t. weights and biases
        # dL/dW = (input)T * (dL/d_Z)
        # dL/dB = Sum of (dL/dZ)
        self.grad_weights = np.transpose(self.input)@grad
        self.grad_bias = np.sum(grad, axis=0)
        return grad_input

    def step(self, step_size):
        self.weights -= step_size * self.grad_weights
        self.bias -= step_size * self.grad_bias


class CLASS_P_CONV:
    def __init__(self, input_dim, output_dim):
        # WOON - need to make sure the rati of weight
        self.weights = np.random.randn(input_dim, output_dim)*4*10
        self.bias = np.ones((1, output_dim))*0.5

    # During the forward pass, we simply compute Xw+b
    def forward(self, input):
        self.input = input
        return self.input@self.weights + self.bias

    def backward(self, grad):
        #   dL/d_input = (dL/d_output) * (d_output/d_input)
        #                          where (d_output/d_input) = wT
        #
        grad_input = grad@np.transpose(self.weights)
        # compute gradient w.r.t. weights and biases
        # dL/dW = (input)T * (dL/d_Z)
        # dL/dB = Sum of (dL/dZ)
        self.grad_weights = np.transpose(self.input)@grad
        self.grad_bias = np.sum(grad, axis=0)
        return grad_input

    def step(self, step_size):
        self.weights -= step_size * self.grad_weights
        self.bias -= step_size * self.grad_bias


class CLASS_AVG_POOLING:
    # Forward pass is max(0,input)
    def forward(self, input):
        return np.sum(input)/np.total_number(input)
    # Backward pass masks out same elements

    def backward(self, grad):
        return grad
    # No parameters so nothing to do during a gradient descent step

    def step(self, step_size):
        return


class CLASS_FULLY_CONNECTED:
    # Forward pass is max(0,input)
    def forward(self, input):
        return  # calculate 64 to 10
    # Backward pass masks out same elements

    def backward(self, grad):
        return grad
    # No parameters so nothing to do during a gradient descent step

    def step(self, step_size):
        return


class CLASS_KWS_NETWORK:
    def __init__(self, input_dim, output_dim, hidden_dim, number_of_hidden_layers, activation="ReLU"):
        self.layers = [CLASS_CONV(input_dim, hidden_dim)]
        self.layers.append(CLASS_ReLU())
        for i in range(number_of_hidden_layers):
            self.layers.append(CLASS_D_CONV(hidden_dim, hidden_dim))
            self.layers.append(CLASS_ReLU())
            self.layers.append(CLASS_P_CONV(hidden_dim, hidden_dim))
            self.layers.append(CLASS_ReLU())
        self.layers.append(CLASS_AVG_POOLING())
        self.layers.append(CLASS_FULLY_CONNECTED())

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, step_size):
        for layer in self.layers:
            layer.step(step_size)

# Rectified Linear Unit Activation Function


class CLASS_ReLU:
    # Forward pass is max(0,input)
    def forward(self, input):
        self.mask = (input > 0)
        return input * self.mask
    # Backward pass masks out same elements

    def backward(self, grad):
        return grad * self.mask
    # No parameters so nothing to do during a gradient descent step

    def step(self, step_size):
        return


#####################################################
# Utility Functions for Computing Loss / Val Metrics
#####################################################
def softmax(x):
    x -= np.max(x, axis=1)[:, np.newaxis]
    return np.exp(x) / (np.sum(np.exp(x), axis=1)[:, np.newaxis])


class CLASS_CrossEntropySoftmax:

    def forward(self, logits, labels):
        self.probs = softmax(logits)
        self.labels = labels
        return -np.mean(np.log(self.probs[np.arange(len(self.probs))[:, np.newaxis], labels]+0.00001))

    def backward(self):
        grad = self.probs
        grad[np.arange(len(self.probs))[:, np.newaxis], self.labels] -= 1

        # Equation #18 first value
        return grad.astype(np.float64)/len(self.probs)


def evaluateValidation(model, X_val, Y_val, batch_size):
    val_loss_running = 0
    val_acc_running = 0
    j = 0

    lossFunc = CLASS_CrossEntropySoftmax()

    while j < len(X_val):
        b = min(batch_size, len(X_val)-j)
        X_batch = X_val[j:j+b]
        Y_batch = Y_val[j:j+b].astype(np.int32)

        logits = model.forward(X_batch)
        loss = lossFunc.forward(logits, Y_batch)
        acc = np.mean(np.argmax(logits, axis=1)[:, np.newaxis] == Y_batch)

        val_loss_running += loss*b
        val_acc_running += acc*b

        j += batch_size

    return val_loss_running/len(X_val), val_acc_running/len(X_val)


#####################################################
# Utility Functions for Loading and Displaying Data
#####################################################
def loadData(normalize=True):
    train = np.loadtxt("kws_train.csv", delimiter=",", dtype=np.float64)
    val = np.loadtxt("kws_val.csv", delimiter=",", dtype=np.float64)
    test = np.loadtxt("kws_test.csv", delimiter=",", dtype=np.float64)

    # Normalize Our Data
    if normalize:
        X_train = train[:, :-1]/256-0.5
        X_val = val[:, :-1]/256-0.5
        X_test = test/256-0.5
    else:
        X_train = train[:, :-1]
        X_val = val[:, :-1]
        X_test = test

    Y_train = train[:, -1].astype(np.int32)[:, np.newaxis]
    Y_val = val[:, -1].astype(np.int32)[:, np.newaxis]

    logging.info("Loaded train: " + str(X_train.shape))
    logging.info("Loaded val: " + str(X_val.shape))
    logging.info("Loaded test: " + str(X_test.shape))

    return X_train, Y_train, X_val, Y_val, X_test


def displayExample(x):
    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
