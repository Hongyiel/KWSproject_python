import numpy as np

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
    # TEST INPUT
    kws_network = CLASS_KWS_NETWORK()
    # start_matrix = np.random.randint(9, size=(49, 11)) + 1
    start_matrix = np.ones((49, 11))
    result_matrix = kws_network.forward(start_matrix)
    print("result matrix[0]: \n", result_matrix[0])
    print("result matrix[1]: \n", result_matrix[1])
    print("result matrix[2]: \n", result_matrix[2])
    print("depth: ", result_matrix.shape[0])
    print("width: ", result_matrix[0].shape[1])
    print("height: ", result_matrix[0].shape[0])


class CLASS_CONV:
    def __init__(self):
        # Array of Kernel Matrix
        self.kernel_count = 64
        self.kernel = np.zeros((self.kernel_count, 10, 4))
        for i in range(self.kernel_count):
            #self.kernel[i] = np.random.randint(9, size=(10, 4)) + 1
            self.kernel[i] = np.ones((10, 4))
        print("kernel matrix[0]: \n", self.kernel[0])

        self.strides = 2
        # self.bias = np.ones( (1,output_dim) )*0.5

    # During the forward pass, we simply compute Xw+b
    def forward(self, input):
        print("CNV")
        # Padding applied
        self.input = np.pad(input, ((4, 4), (1, 1)),
                            'constant', constant_values=0)
        print("Padded matrix: \n", self.input)
        print("width: ", self.input.shape[1])
        print("height: ", self.input.shape[0])

        # Gather Shapes of Kernel + Image + Padding
        kernel_width = 4
        kernel_height = 10
        input_height = self.input.shape[0]
        input_width = self.input.shape[1]

        # Shape of Output Convolution
        output_width = 6
        output_height = 25
        output_matrix = np.zeros(
            (self.kernel_count, output_height, output_width))

        # Iterate through image
        for i in range(self.kernel_count):
            for y in range(input_height):
                # Exit Convolution
                if y > input_height - kernel_height:
                    break
                # Only Convolve if y has gone down by the specified Strides
                if y % self.strides == 0:
                    for x in range(input_width):
                        # Go to next row once kernel is out of bounds
                        if x > input_width - kernel_width:
                            break

                        # Only Convolve if x has moved by the specified Strides
                        if x % self.strides == 0:
                            try:
                                output_y = int(np.floor(y/self.strides))
                                output_x = int(np.floor(x/self.strides))
                                output_matrix[i][output_y, output_x] = (
                                    self.kernel[i] * self.input[y: y + kernel_height, x: x + kernel_width]).sum()
                            except:
                                break

        return output_matrix

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
    def __init__(self):
        # Array of Kernel Matrix
        self.kernel_count = 64
        self.kernel = np.zeros((self.kernel_count, 3, 3))
        for i in range(self.kernel_count):
            #self.kernel[i] = np.random.randint(9, size=(3, 3)) + 1
            self.kernel[i] = np.ones((3, 3))
        print("kernel matrix[0]: \n", self.kernel[0])

        self.strides = 1
        # self.bias = np.ones( (1,output_dim) )*0.5

    # During the forward pass, we simply compute Xw+b
    def forward(self, input):
        print("D-CNV")
        print("Input: \n", input[0])
        # Padding applied
        self.input = np.zeros((input.shape[0], 27, 8))
        for i in range(input.shape[0]):
            self.input[i] = np.pad(
                input[i], ((1, 1), (1, 1)), 'constant', constant_values=0)
        print("Padded matrix: \n", self.input[0])
        print("width: ", self.input[0].shape[1])
        print("height: ", self.input[0].shape[0])

        # Gather Shapes of Kernel + Image + Padding
        kernel_width = 3
        kernel_height = 3
        input_width = self.input[0].shape[1]
        input_height = self.input[0].shape[0]

        # Shape of Output Convolution
        output_width = 6
        output_height = 25
        output_matrix = np.zeros(
            (self.kernel_count, output_height, output_width))

        # Iterate through image
        for i in range(self.kernel_count):
            for y in range(input_height):
                # Exit Convolution
                if y > input_height - kernel_height:
                    break
                # Only Convolve if y has gone down by the specified Strides
                if y % self.strides == 0:
                    for x in range(input_width):
                        # Go to next row once kernel is out of bounds
                        if x > input_width - kernel_width:
                            break

                        # Only Convolve if x has moved by the specified Strides
                        if x % self.strides == 0:
                            try:
                                output_y = int(np.floor(y/self.strides))
                                output_x = int(np.floor(x/self.strides))
                                output_matrix[i][output_y, output_x] = (
                                    self.kernel[i] * self.input[i][y: y + kernel_height, x: x + kernel_width]).sum()
                            except:
                                break

        return output_matrix

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
    def __init__(self):
        # Array of Kernel Matrix
        self.kernel_count = 64
        self.kernel = np.zeros((self.kernel_count, 64))
        for i in range(self.kernel_count):
            #self.kernel[i] = np.random.randint(9, size=(3, 3)) + 1
            self.kernel[i] = 1
        print("kernel matrix[0]: \n", self.kernel[0])

        self.strides = 1
        # self.bias = np.ones( (1,output_dim) )*0.5

    # During the forward pass, we simply compute Xw+b
    def forward(self, input):
        print("P-CNV")
        print("P_cnn: Input: \n", input[0])
        
        # Padding applied
        self.input = np.zeros((input.shape[0], 27, 8))
        for i in range(input.shape[0]):
            self.input[i] = np.pad(
                input[i], ((1, 1), (1, 1)), 'constant', constant_values=0)
        print("Padded matrix: \n", self.input[0])
        print("width: ", self.input[0].shape[1])
        print("height: ", self.input[0].shape[0])

        # Gather Shapes of Kernel + Image + Padding
        # Pointwise changed
        kernel_width = 1
        kernel_height = 1
        
        
        input_width = self.input[0].shape[1]
        input_height = self.input[0].shape[0]

        # Shape of Output Convolution
        output_width = 6
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
                if y > input_height - kernel_height:
                    break
                # Only Convolve if y has gone down by the specified Strides
                if y % self.strides == 0:
                    for x in range(input_width):
                        # Go to next row once kernel is out of bounds
                        if x > input_width - kernel_width:
                            break

                        # Only Convolve if x has moved by the specified Strides
                        if x % self.strides == 0:
                            try:
                                output_y = int(np.floor(y/self.strides))
                                output_x = int(np.floor(x/self.strides))
                                #  0 1 2 3 ... 
                                
                                # calculation part
                                for j in range(input.shape[0]):
                                  output_matrix[i][output_y, output_x] += input[j][y,x] * self.kernel[i][j]
                                  
                            except:
                                break

        return output_matrix

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
    def __init__(self):
        self.layers = [CLASS_CONV()]
        self.layers.append(CLASS_ReLU())
        self.layers.append(CLASS_D_CONV())
        self.layers.append(CLASS_ReLU())
        
        self.layers.append(CLASS_P_CONV())
        self.layers.append(CLASS_ReLU())
        """
    for i in range(number_of_hidden_layers):
      self.layers.append(CLASS_D_CONV(hidden_dim, hidden_dim))
      self.layers.append(CLASS_ReLU())
      self.layers.append(CLASS_P_CONV(hidden_dim, hidden_dim))
      self.layers.append(CLASS_ReLU())
    self.layers.append(CLASS_AVG_POOLING())
    self.layers.append(CLASS_FULLY_CONNECTED()) 
    """

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


if __name__ == "__main__":
    main()
