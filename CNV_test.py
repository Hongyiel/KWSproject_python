import numpy as np
import os.path
import sys
import librosa  # for audio related library using
import csv

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
# np.random.seed(102)
step_size = 0.001
batch_size = 10000
max_epochs = 1000
np.set_printoptions(precision=3)

# GLOBAL PARAMETERS FOR NETWORK ARCHITECTURE
# input 1:    11x49x1 --> conv 4x10/2                   --> 6x25x64
# layer 2~5:  6x25x64 --> dw_conv 3x3/1, pw_conv 1x1/1  --> 6x25x64
# layer 6:   6x25x64 --> grand average                 --> 1x1x64
# output:     1x1x64  --> fully_connected               --> 1x1x10
number_of_hidden_layers = 4
number_of_hidden_lnodes = 64  # only for layer > 1
activation = "ReLU"

#################### Audio Data Area ####################

wav_file = "go_nohash_0.wav"
n_mfcc = 49
n_mels = 10
n_fft = 640  # 512
win_length = 640  # 160
hop_length = 320  # 160
fmin = 20
fmax = 4000
sr = 16000
# how to call data from librosa
# y, sr = librosa.load(librosa.ex('trumpet'))
audio_data, sr = librosa.load(wav_file, sr=sr)  # , offset=0.04, duration=1.0)
print('---audio_sr:', sr)
print('---audio_data:', audio_data.shape)
audio_np = np.array(audio_data, np.float32)
print('audio_np:', audio_np.shape)

mfcc_librosa = librosa.feature.mfcc(y=audio_data, sr=sr,
                                    win_length=win_length, hop_length=hop_length,
                                    # it will be start FFT from begins at y[t* hop_length]
                                    center=False,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc, n_mels=n_mels,
                                    fmin=fmin, fmax=fmax, htk=False
                                    )
# mfcc_librosa is the data that consist of 49 x 10 matrix
print('mfcc_librosa', mfcc_librosa.shape)

# If data directory is "dog" then get 0 
# If data directory is "cat" tjem get 1
# ...
# this is how the data set from training file(Answer sheet)


##########################################################


def main():
    # TEST INPUT
    wave_data, label = loadData()
    kws_network = CLASS_KWS_NETWORK()
    lossFunc = CLASS_CrossEntropySoftmax()

    start_matrix = np.transpose(mfcc_librosa)

    # Compute the scores for our 10 classes using our model
    result = kws_network.forward(start_matrix)
    # traindata = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # loss = lossFunc.forward(result[0], traindata)
    # # accuracy
    # acc = np.mean(np.argmax(result, axis=1)[:, np.newaxis] == traindata)
    # # Compute gradient of Cross-Entropy Loss with respect to logits
    # loss_grad = lossFunc.backward()
    # # Pass gradient back through networks
    # kws_network.backward(loss_grad)
    # # Take a step of gradient descent
    # kws_network.step(step_size)

    print("\n\n\n-----------RESULT-----------")
    print("Result: \n", result)


class CLASS_CONV:
    def __init__(self):
        # Initialize parameters
        self.kernel_count = 64
        self.kernel_width = 4
        self.kernel_height = 10
        self.strides = 2

        # # Initialize the kernel array
        # self.kernel = np.zeros(
        #     (self.kernel_count, self.kernel_height, self.kernel_width))
        # # Assign random filter value
        # for i in range(self.kernel_count):
        #     self.kernel[i] = np.random.randn(
        #         self.kernel_height, self.kernel_width)
        self.kernel = np.random.randn(self.kernel_count, self.kernel_height, self.kernel_width)

        # Print function
        print("CNV kernel matrix[0]: \n", self.kernel[0])

        self.bias = np.ones((self.kernel_count, 1, 1))*0.5

    def forward(self, input):
        print("\n\n\n-----------CNV-----------")
        # Padding applied
        self.input = np.pad(input, ((4, 5), (1, 1)),
                            'constant', constant_values=0)

        # Print function (Suppressed normally due to size)
        #print("Input matrix: \n", self.input)
        #print("width: ", self.input.shape[1])
        #print("height: ", self.input.shape[0])

        # Input information
        input_width = self.input.shape[1]
        input_height = self.input.shape[0]

        # Output information
        output_width = 5
        output_height = 25
        output_matrix = np.zeros(
            (self.kernel_count, output_height, output_width))

        # Iterate through image
        for i in range(self.kernel_count):
            for y in range(input_height):
                # Exit Convolution
                if y > input_height - self.kernel_height:
                    break
                # Only Convolve if y has gone down by the specified Strides
                if y % self.strides == 0:
                    for x in range(input_width):
                        # Go to next row once kernel is out of bounds
                        if x > input_width - self.kernel_width:
                            break

                        # Only Convolve if x has moved by the specified Strides
                        if x % self.strides == 0:
                            try:
                                output_y = int(np.floor(y/self.strides))
                                output_x = int(np.floor(x/self.strides))
                                output_matrix[i][output_y, output_x] = (
                                    self.kernel[i] * self.input[y: y + self.kernel_height, x: x + self.kernel_width]).sum()
                            except:
                                break

        print("\nOutput matrix[0]: \n", output_matrix[0])
        print("width: ", output_matrix[0].shape[1])
        print("height: ", output_matrix[0].shape[0])
        return output_matrix + self.bias

    def backward(self, grad):
        #   dL/d_input = (dL/d_output) * (d_output/d_input)
        #                          where (d_output/d_input) = wT
        #
        grad_input = grad@np.transpose(self.kernel)
        # compute gradient w.r.t. weights and biases
        # dL/dW = (input)T * (dL/d_Z)
        # dL/dB = Sum of (dL/dZ)
        self.grad_weights = np.transpose(self.input)@grad
        self.grad_bias = np.sum(grad, axis=0)
        return grad_input

    def step(self, step_size):
        self.kernel -= step_size * self.grad_weights
        self.bias -= step_size * self.grad_bias


class CLASS_D_CONV:
    def __init__(self):
        # Initialize parameters
        self.kernel_count = 64
        self.kernel_width = 3
        self.kernel_height = 3
        self.strides = 1

        # # Initialize the kernel array
        # self.kernel = np.zeros(
        #     (self.kernel_count, self.kernel_height, self.kernel_width))
        # # Assign random filter value
        # for i in range(self.kernel_count):
        #     self.kernel[i] = np.random.randn(
        #         self.kernel_height, self.kernel_width)

        self.kernel = np.random.randn(self.kernel_count, self.kernel_height, self.kernel_width)

        # Print Function
        print("D-CNV kernel matrix[0]: \n", self.kernel[0])

        self.bias = np.ones((self.kernel_count, 1, 1))*0.5

    def forward(self, input):
        print("\n\n\n-----------D-CNV-----------")
        # Padding applied
        self.input = np.zeros((input.shape[0], 27, 7))
        for i in range(input.shape[0]):
            self.input[i] = np.pad(
                input[i], ((1, 1), (1, 1)), 'constant', constant_values=0)

        # Print function
        print("Input matrix: \n", self.input[0])
        print("width: ", self.input[0].shape[1])
        print("height: ", self.input[0].shape[0])

        # Input information
        input_width = self.input.shape[1]
        input_height = self.input.shape[0]

        # Output information
        output_width = 5
        output_height = 25
        output_matrix = np.zeros(
            (self.kernel_count, output_height, output_width))

        # Iterate through image
        for i in range(self.kernel_count):
            for y in range(input_height):
                # Exit Convolution
                if y > input_height - self.kernel_height:
                    break
                # Only Convolve if y has gone down by the specified Strides
                if y % self.strides == 0:
                    for x in range(input_width):
                        # Go to next row once kernel is out of bounds
                        if x > input_width - self.kernel_width:
                            break

                        # Only Convolve if x has moved by the specified Strides
                        if x % self.strides == 0:
                            try:
                                output_y = int(np.floor(y/self.strides))
                                output_x = int(np.floor(x/self.strides))
                                output_matrix[i][output_y, output_x] = (
                                    self.kernel[i] * self.input[i][y: y + self.kernel_height, x: x + self.kernel_width]).sum()
                            except:
                                break

        print("\nOutput matrix[0]: \n", output_matrix[0])
        print("width: ", output_matrix[0].shape[1])
        print("height: ", output_matrix[0].shape[0])
        return output_matrix + self.bias

    def backward(self, grad):
        #   dL/d_input = (dL/d_output) * (d_output/d_input)
        #                          where (d_output/d_input) = wT
        #
        grad_input = grad@np.transpose(self.kernel)
        # compute gradient w.r.t. weights and biases
        # dL/dW = (input)T * (dL/d_Z)
        # dL/dB = Sum of (dL/dZ)
        self.grad_weights = np.transpose(self.input)@grad
        self.grad_bias = np.sum(grad, axis=0)
        return grad_input

    def step(self, step_size):
        self.kernel -= step_size * self.grad_weights
        self.bias -= step_size * self.grad_bias


class CLASS_P_CONV:
    def __init__(self):
        # Initialize parameters
        self.kernel_count = 64
        self.kernel_width = 1
        self.kernel_height = 1
        self.strides = 1

        # Initialize the kernel array
        #self.kernel = np.zeros((self.kernel_count, 64))

        # Assign random filter value
        # for i in range(self.kernel_count):
        #     self.kernel[i] = np.random.randn(1)
        self.kernel = np.random.randn(self.kernel_count, 64)

        # Print function
        print("P-CNV kernel matrix[0]: \n", self.kernel[0])

        self.bias = np.ones((self.kernel_count, 1, 1))*0.5

    def forward(self, input):
        print("\n\n\n-----------P-CNV-----------")

        self.input = np.zeros((input.shape[0], 25, 5))
        for i in range(input.shape[0]):
            self.input[i] = input[i]

        # Print function
        print("Input matrix: \n", self.input[0])
        print("width: ", self.input[0].shape[1])
        print("height: ", self.input[0].shape[0])

        # Input information
        input_width = self.input.shape[1]
        input_height = self.input.shape[0]

        # Output information
        output_width = 5
        output_height = 25
        output_matrix = np.zeros(
            (self.kernel_count, output_height, output_width))

        # Iterate through image
        # i is first layer
        # i is number of kernel
        for i in range(self.kernel_count):
          # kernel 64
            for y in range(input_height):
                # Exit Convolution
                if y > input_height - self.kernel_height:
                    break
                # Only Convolve if y has gone down by the specified Strides
                if y % self.strides == 0:
                    for x in range(input_width):
                        # Go to next row once kernel is out of bounds
                        if x > input_width - self.kernel_width:
                            break

                        # Only Convolve if x has moved by the specified Strides
                        if x % self.strides == 0:
                            try:
                                output_y = int(np.floor(y/self.strides))
                                output_x = int(np.floor(x/self.strides))
                                #  0 1 2 3 ...

                                # calculation part
                                for j in range(input.shape[0]):
                                    output_matrix[i][output_y,
                                                     output_x] += self.input[j][y, x] * self.kernel[i][j]

                            except:
                                break

        print("\nOutput matrix[0]: \n", output_matrix[0])
        print("width: ", output_matrix[0].shape[1])
        print("height: ", output_matrix[0].shape[0])
        return output_matrix + self.bias

    def backward(self, grad):
        #   dL/d_input = (dL/d_output) * (d_output/d_input)
        #                          where (d_output/d_input) = wT
        #
        grad_input = grad@np.transpose(self.kernel)
        # compute gradient w.r.t. weights and biases
        # dL/dW = (input)T * (dL/d_Z)
        # dL/dB = Sum of (dL/dZ)
        self.grad_weights = np.transpose(self.input)@grad
        self.grad_bias = np.sum(grad, axis=0)
        return grad_input

    def step(self, step_size):
        self.kernel -= step_size * self.grad_weights
        self.bias -= step_size * self.grad_bias


class CLASS_AVG_POOLING:
    # Forward pass is max(0,input)
    def forward(self, input):
        print("\n\n\n-----------AVG POOLING-----------")
        # 5 x 25
        input_avg_pool = np.zeros(64)
        for i in range(input.shape[0]):
            input_avg_pool[i] = (np.sum(input[i])/np.size(input[i]))

        print("AVG POOLING: ", input_avg_pool)
        return input_avg_pool
    # Backward pass masks out same elements

    def backward(self, grad):
        return grad
    # No parameters so nothing to do during a gradient descent step

    def step(self, step_size):
        return


class CLASS_FULLY_CONNECTED:
    def __init__(self):
        self.kernel = np.random.rand(64, 12)
        self.bias = np.random.rand(1, 12)
    # Forward pass is max(0,input)

    def forward(self, input):
        self.input = input
        # calculate 64 to 10
        return np.dot(self.input, self.kernel) + self.bias
    # Backward pass masks out same elements

    def backward(self, grad):
        return grad
    # No parameters so nothing to do during a gradient descent step

    def step(self, step_size):
        return


class CLASS_KWS_NETWORK:
    def __init__(self):
        self.layers = [CLASS_CONV()]
        self.layers.append(CLASS_ReLU())
        for i in range(number_of_hidden_layers):
          self.layers.append(CLASS_D_CONV())
          self.layers.append(CLASS_ReLU())
          self.layers.append(CLASS_P_CONV())
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

def loadData():
    wav_file = np.loadtxt("wav_data.csv", delimiter=",", dtype=np.float64)
    wav_train = wav_file[:,:-1]    
    print("wav_file shape:", wav_file.shape)    
    print("wav_train shape:", wav_train.shape)

    arr = []
    label = []    
    for line in wav_train:
        wav_arr = np.array(line).reshape(10,49)        
        arr.append(wav_arr)
        label.append(line[[-1]])
    print("wav_arr shape:", wav_arr.shape)
    print("arr shape:", np.array(arr).shape)
    print("label shape:", np.array(label).shape)

    return arr, label

if __name__ == "__main__":
    main()
