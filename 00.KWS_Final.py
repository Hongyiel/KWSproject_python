import numpy as np
import os.path
import sys
import librosa  # for audio related library using
import csv

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
# np.random.seed(102)
step_size = 0.001
global batch_size
batch_size = 3
max_epochs = 1000
np.set_printoptions(precision=3)

# GLOBAL PARAMETERS FOR NETWORK ARCHITECTURE
# input 1:    11x49x1 --> conv 4x10/2                   --> 6x25x64
# layer 2~5:  6x25x64 --> dw_conv 3x3/1, pw_conv 1x1/1  --> 6x25x64
# layer 6:   6x25x64 --> grand average                 --> 1x1x64
# output:     1x1x64  --> fully_connected               --> 1x1x10
number_of_hidden_layers = 1
number_of_hidden_lnodes = 64  # only for layer > 1
activation = "ReLU"

# #################### Audio Data Area ####################

# wav_file = "go_nohash_0.wav"
# n_mfcc = 49
# n_mels = 10
# n_fft = 640  # 512
# win_length = 640  # 160
# hop_length = 320  # 160
# fmin = 20
# fmax = 4000
# sr = 16000
# # how to call data from librosa
# # y, sr = librosa.load(librosa.ex('trumpet'))
# audio_data, sr = librosa.load(wav_file, sr=sr)  # , offset=0.04, duration=1.0)
# print('---audio_sr:', sr)
# print('---audio_data:', audio_data.shape)
# audio_np = np.array(audio_data, np.float32)
# print('audio_np:', audio_np.shape)

# mfcc_librosa = librosa.feature.mfcc(y=audio_data, sr=sr,
#                                     win_length=win_length, hop_length=hop_length,
#                                     # it will be start FFT from begins at y[t* hop_length]
#                                     center=False,
#                                     n_fft=n_fft,
#                                     n_mfcc=n_mfcc, n_mels=n_mels,
#                                     fmin=fmin, fmax=fmax, htk=False
#                                     )
# # mfcc_librosa is the data that consist of 49 x 10 matrix
# print('mfcc_librosa', mfcc_librosa.shape)

# # If data directory is "dog" then get 0
# # If data directory is "cat" tjem get 1
# # ...
# # this is how the data set from training file(Answer sheet)


# ##########################################################
constant_batch_size = 3


def main():
    
    # TEST INPUT
    wav_file = np.loadtxt("wav_data.csv", delimiter=",", dtype=np.float32)
    ## wav_file = np.transpose(wav_49x10)
    wav_train = wav_file[:, :-1]
    wav_label = wav_file[:, -1]
    
    shuffle = np.random.permutation(len(wav_label))

    wav_train = wav_train[shuffle]
    wav_label = wav_label[shuffle]
    

    print("wav_train:\n", wav_train)
    print("wav_label:\n", wav_label)

    
    print("wav_file shape:", wav_file.shape)
    print("wav_train shape:", wav_train.shape)
    print("\n\n\n")

    kws_network = CLASS_KWS_NETWORK()
    lossFunc = CLASS_CrossEntropySoftmax()

    # Add for loop for epoch later here 
    
    index = 0

    while(index < wav_train.shape[0]):
        if index + constant_batch_size > wav_train.shape[0]:
            end = index + (wav_train.shape[0] % constant_batch_size)
        else:
            end = index + constant_batch_size
        print("this is end: ", end)
        print("This is index: ", index)
        x_train = wav_train[index : end]
        
        
        x_train = x_train.reshape(end - index, 49, 10)
        y_train = wav_label[index : end]
        index += constant_batch_size
        batch_size = end - index
        # print("batch_size ", batch_size)
        # print("x_train:", x_train)
        # print("y_train: ", y_train)

        # Compute the scores for our 10 classes using our model
        result = kws_network.forward(x_train)

        print("-----------RESULT-----------")   
        print("Result: \n", result)

        loss = lossFunc.forward(result, y_train)
        # accuracy
        acc = np.mean(np.argmax(result, axis=1)[:, np.newaxis] == y_train)
        # Compute gradient of Cross-Entropy Loss with respect to logits
        loss_grad = lossFunc.backward()
        print("loss grad: ", loss_grad)
        # Pass gradient back through networks
        kws_network.backward(loss_grad)
        # Take a step of gradient descent
        kws_network.step(step_size)

        
        print("\n\n\n")


class CLASS_CONV:
    def __init__(self,IN,IH,IW,PHu,PHd,PWl,PWr,KH,KW,ST,ON):
        # Initialize parameters
        self.IN = IN    # Input number
        self.IH = IH    # input height
        self.IW = IW    # input width
        self.PHu = PHu  # padding upper
        self.PHd = PHd  # padding bottom 
        self.PWl = PWl  # padding left
        self.PWr = PWr  # padding right

        self.ON= ON     # output numbers

        self.KN = IN*ON # kernel numbers
        self.KH = KH    # kernel height
        self.KW = KW    # kernel width
        self.ST = ST    # stride

        # # Initialize the kernel array -----------------   NCHW
        # self.kernel = np.zeros(
        #     (self.kernel_count, self.kernel_height, self.kernel_width))
        # # Assign random filter value
        # for i in range(self.kernel_count):
        #     self.kernel[i] = np.random.randn(
        #         self.kernel_height, self.kernel_width)
        self.kernel = np.random.randn(self.ON, self.KH, self.KW)
        # self.kernel = np.random.randn(self.ON, self.IN, self.KH, self.KW)

        # Print function
        # print("CNV kernel matrix[0]: \n", self.kernel[0])
        self.bias = np.ones((self.ON))*0.5

    def forward(self, input):
        self.input = input
        # print("\n\n\n-----------CNV-----------")
        # for B in range(batch_size):
        #     padded_input = np.pad(input, ( (0,0), (self.PHu, self.PHd), (self.PWl, self.PWr) ))
        #     # print("This is paddddddddd: ", padded_input)
        #     # padded_input = np.pad(input, ( (self.PHu, self.PHd), (self.PWl, self.PWr) ),
        #     #                     'constant', constant_values=0)
        padded_input = np.pad(input, ( (0,0), (self.PHu, self.PHd), (self.PWl, self.PWr) ))

        # print("+++++++++++++++++++++++++++++++++++++++++")
        # print(padded_input)
        # print("+++++++++++++++++++++++++++++++++++++++++")
        # Input information
        OH = 1 + (self.IH + self.PHu + self.PHd - self.KH)//self.ST
        OW = 1 + (self.IW + self.PWl + self.PWr - self.KW)//self.ST
        S = self.ST # stride?
        n_H = 50 ## self.ST + (self.IH + self.PHu + self.PHd - self.KH)  # 51
        n_W = 10 ## self.ST + (self.IW + self.PWl + self.PWr - self.KW)  # 11
        # print("CONV OH: ",OH)
        # print("CONV OW: ",OW)
        # print("CONV n_H: ",n_H)
        # print("CONV n_W: ",n_W)
        # print("CONV self.ON: ",self.ON)
        output = np.zeros((batch_size,self.ON, OH, OW))
        # Iterate through image
        # output numbers : ON
        for B in range(batch_size):
            for M in range(self.ON):
                # for N in range(self.IN):
                for r in range(0,n_H,S):
                    for c in range(0,n_W,S):
                        output[B,M,r//S,c//S] = np.sum(padded_input[B,r:r+self.KH,c:c+self.KW] * self.kernel[M,:,:]) + self.bias[M]
                    #print("#######################################")
                    #print(padded_input[r:r+self.KH,c:c+self.KW])
                    #print(self.kernel[M,:,:])
                    #print(self.bias[M])
                    #print(padded_input[r:r+self.KH,c:c+self.KW] * self.kernel[M,:,:])
                    #print("M: ",M)
                    #print("r: ",r)
                    #print("c: ",c)
                    #print("---------------------------------------")
                    #print("r//S: ",r//S)
                    #print("c//S: ",c//S)
                    #print(output[M,r//S,c//S])
                    #print("#######################################")
        #print("-----------------------------------")
        #print("forward output at CONV:", output)
        #print("-----------------------------------")
        return output

    def backward(self, grad):
        ## memory allocation for each deviation
        OH = 1 + (self.IH + self.PHu + self.PHd - self.KH)//self.ST
        OW = 1 + (self.IW + self.PWl + self.PWr - self.KW)//self.ST
        S = self.ST
        #IN 49x10 
        padded_input = np.pad(self.input, ( (0,0),(self.PHu, self.PHd), (self.PWl, self.PWr)),'constant', constant_values=0)
        dx = np.zeros(padded_input.shape)
        grad_weights = np.zeros(self.kernel.shape)
        grad_bias = np.zeros(self.bias.shape)
        n_H = 50
        n_W = 10
        #grad_input = grad@np.transpose(self.kernel)
        # for batch operation, it need to add one more dimension in the loop
        for B in range(batch_size):
            for M in range(self.ON):
                for r in range(0,n_H,S): # width of each feature
                    for c in range(0,n_W,S):
                        # print("#######################################")
                        # print("M: ",M)
                        # print("r: ",r)
                        # print("c: ",c)
                        # print("---------------------------------------")
                        # print("r//S: ",r//S)
                        # print("c//S: ",c//S)
                        # print("dx:",dx[r:r+self.KH,c:c+self.KW])
                        # print("grad: ", grad[M,r//S,c//S])
                        # print("kernel:", self.kernel[M,:,:])
                        # print("#######################################")
                        dx[B,r:r+self.KH,c:c+self.KW] += grad[B,M,r//S,c//S] * self.kernel[M,:,:]
        delete_rows = range(self.PHu+self.PHd)
        delete_colm = range(self.PWl+self.PWr)
        dx         = np.delete(dx, delete_rows, axis=1) # ? woon
        grad_input = np.delete(dx, delete_colm, axis=2) # ? woon
        # compute gradient w.r.t. weights and biases
        # dL/dW = (input)T * (dL/d_Z)
        # dL/dB = Sum of (dL/dZ)
        # self.grad_weights = np.transpose(self.input)@grad
        for B in range(batch_size):
            for M in range(self.ON):
                for r in range(OH):
                    for c in range(OW):
                        grad_weights[M,:,:] += grad[B,M, r, c] * padded_input[B,r*S:r*S+self.KH,c*S:c*S+self.KW]
        for B in range(batch_size):
            for M in range(self.ON):
                grad_bias[M] = np.sum(grad[B,M,:,:])

            self.grad_weights = grad_weights
            self.grad_bias = grad_bias
        return grad_input

    def step(self, step_size):
        self.kernel -= step_size * self.grad_weights
        self.bias -= step_size * self.grad_bias


class CLASS_D_CONV:
    def __init__(self,IN,IH,IW,PHu,PHd,PWl,PWr,KW,KH,ST,ON):
        # Initialize parameters
        self.IN = IN
        self.IH = IH
        self.IW = IW
        self.PHu = PHu
        self.PHd = PHd
        self.PWl = PWl
        self.PWr = PWr
        self.ON = ON
        self.KN = IN
        self.KW = KW
        self.KH = KH
        self.ST = ST

        # # Initialize the kernel array
        # self.kernel = np.zeros(
        #     (self.kernel_count, self.kernel_height, self.kernel_width))
        # # Assign random filter value
        # for i in range(self.kernel_count):
        #     self.kernel[i] = np.random.randn(
        #         self.kernel_height, self.kernel_width)

        self.kernel = np.random.randn(self.KN, self.KH, self.KW)

        # Print Function
        # print("D-CNV kernel matrix[0]: \n", self.kernel[0])

        self.bias = np.ones((self.KN, 1, 1))*0.5

    def forward(self, input):
        # print("\n\n\n-----------Depth wise Conv-----------")
        self.input = input
        
        padded_input = np.pad(input, ( (0,0), (0,0), (self.PHu, self.PHd), (self.PWl, self.PWr) ),
                            'constant', constant_values=0)
        # Input information
        OH = 1 + (self.IH + self.PHu + self.PHd - self.KH)//self.ST
        OW = 1 + (self.IW + self.PWl + self.PWr - self.KW)//self.ST
        S = self.ST
        output = np.zeros((batch_size,self.ON, OH, OW))
        # for batch operation, it need to add one more dimension in the loop
        # Iterate through image
        for B in range(batch_size):
            for D in range(self.IN):
                for r in range(0,self.IH,S):
                    for c in range(0,self.IW,S):
                        output[B,D,r//S,c//S] = np.sum(padded_input[B,D,r:r+self.KH,c:c+self.KW] * self.kernel[D,:,:]) + self.bias[D]
                        # print("#######################################")
                        # print(padded_input[D,r:r+self.KH,c:c+self.KW])
                        # print(self.kernel[D,:,:])
                        # print(self.bias[D])
                        # #print(padded_input[D,r:r+self.KH,c:c+self.KW] * self.kernel[D,:,:])
                        # print("D: ",D)
                        # print("r: ",r)
                        # print("c: ",c)
                        # print("---------------------------------------")
                        # print("r//S: ",r//S)
                        # print("c//S: ",c//S)
                        # print(output[D,r//S,c//S])
                        # print("#######################################")
        #print("-----------------------------------")
        #print("forward output at CONV:", output)
        #print("-----------------------------------")                              
        #print("forward output at CONV:", output)
        return output

    def backward(self, grad):
        ## memory allocation for each deviation
        OH = 1 + (self.IH + self.PHu + self.PHd - self.KH)//self.ST
        OW = 1 + (self.IW + self.PWl + self.PWr - self.KW)//self.ST
        S = self.ST
        padded_input = np.pad(self.input, ((0,0),(0,0), (self.PHu, self.PHd), (self.PWl, self.PWr)),'constant', constant_values=0)
        print("padddedmidjeijd:: ", np.shape(padded_input))
        dx = np.zeros(padded_input.shape)
        print("grad shape: ",np.shape(grad))
        grad_weights = np.zeros(self.kernel.shape)
        grad_bias = np.zeros(self.bias.shape)
        #   dL/d_input = (dL/d_output) * (d_output/d_input)
        #                          where (d_output/d_input) = wT
        #
        #grad_input = grad@np.transpose(self.kernel)
        # for batch operation, it need to add one more dimension in the loop
        
        for B in range(batch_size):
            for D in range(self.ON): # number of Feature of INPUT
                for r in range(0,self.IH,S): # width of each feature
                    for c in range(0,self.IW,S):
                        # print("c: ", c)
                        
                        dx[B,D,r:r+self.KH,c:c+self.KW] += grad[B,D,r//S,c//S] * self.kernel[D,:,:]
                        
              
                        # print("dx---------------------------------------- :\n",dx[B,D,r:r+self.KH,c:c+self.KW])
                # print("dx: ",dx.shape)
        delete_rows = range(self.PHu+self.PHd)
        delete_colm = range(self.PWl+self.PWr)
        dx         = np.delete(dx, delete_rows, axis=2) # ? woon
        grad_input = np.delete(dx, delete_colm, axis=3) # ? woon
                # print("grad_input: ",grad_input.shape)
                # compute gradient w.r.t. weights and biases
                # dL/dW = (input)T * (dL/d_Z)
                # dL/dB = Sum of (dL/dZ)
                # self.grad_weights = np.transpose(self.input)@grad
        for B in range(batch_size):
            for D in range(self.ON):
                for r in range(OH):
                    for c in range(OW):
                        grad_weights[D,:,:] += grad[B,D, r, c] * padded_input[B,D,r*S:r*S+self.KH,c*S:c*S+self.KW]
        for B in range(batch_size):
            for D in range(self.ON):
                grad_bias[D] = np.sum(grad[B,D,:,:])

            self.grad_weights = grad_weights
            self.grad_bias = grad_bias
        return grad_input

    def step(self, step_size):
        self.kernel -= step_size * self.grad_weights
        self.bias -= step_size * self.grad_bias



class CLASS_P_CONV:
    def __init__(self,IN,IH,IW,KW,KH,ON):
        # Initialize parameters
        self.IN = IN
        self.IH = IH
        self.IW = IW
        self.ON = ON
        self.KN = IN*ON
        self.KW = KW
        self.KH = KH

        # Initialize the kernel array
        #self.kernel = np.zeros((self.kernel_count, 64))

        # Assign random filter value
        # for i in range(self.kernel_count):
        #     self.kernel[i] = np.random.randn(1)
        self.kernel = np.random.randn(self.KN, self.KH, self.KW)
        self.kernel = self.kernel.reshape(self.ON,self.IN)

        # Print function
        # print("P-CNV kernel matrix[0]: \n", self.kernel[0])

        self.bias = np.ones((self.ON, 1, 1))*0.5

    def forward(self, input):
        # print("\n\n\n-----------Point wise Conv-----------")
        # Input information
        self.input = input
        OH = self.IH
        OW = self.IW
        output = np.zeros((batch_size,self.ON, OH, OW))
        # for batch operation, it need to add one more dimension in the loop
        # Iterate through image
        # print("ON: ", self.ON)
        # print("IN: ", self.IN)
        for B in range(batch_size):
            for M in range(self.ON):
                for N in range(self.IN):
                    for r in range(0,self.IH):
                        for c in range(0,self.IW):
                            # print("#######################################")
                            # print("M: ",M)
                            # print("N: ",N)
                            # print("r: ",r)
                            # print("c: ",c)
                            # print("out r: ",r)
                            # print("out c: ",c)
                            # print("---------------------------------------")
                            # print(input[N,r,c])
                            # print(self.kernel[M,N])
                            # print(self.bias[M])
                            # print(output[M,r,c])
                            # print("#######################################")
                            output[B,M,r,c] += np.sum(self.input[B,N,r,c] * self.kernel[M,N]) + self.bias[M]
        #print("forward output at Point wise CONV:", output)
        return output

    def backward(self, grad):
        ## memory allocation for each deviation
        OH = self.IH
        OW = self.IW
        dx = np.zeros(self.input.shape)
        grad_weights = np.zeros(self.kernel.shape)
        grad_bias = np.zeros(self.bias.shape)
        #   dL/d_input = (dL/d_output) * (d_output/d_input)
        #                          where (d_output/d_input) = wT
        #
        #grad_input = grad@np.transpose(self.kernel)
        # for batch operation, it need to add one more dimension in the loop
        for B in range(batch_size):
            for M in range(self.ON): #
                for N in range(self.IN): #
                    for r in range(0,self.IH): # 
                        for c in range(0,self.IW):
                            dx[B,N,r,c] += grad[B,M,r,c] * self.kernel[M,N]
        grad_input = dx
        # compute gradient w.r.t. weights and biases
        # dL/dW = (input)T * (dL/d_Z)
        # dL/dB = Sum of (dL/dZ)
        # self.grad_weights = np.transpose(self.input)@grad
        for B in range(batch_size):
            for M in range(self.ON):
                for N in range(self.IN):
                    for r in range(OH):
                        for c in range(OW):
                            grad_weights[M,N] += grad[B,M, r, c] * self.input[B,N,r,c]
        for B in range(batch_size):
            for M in range(self.ON):
                grad_bias[M] = np.sum(grad[B,M,:,:])

        self.grad_weights = grad_weights
        self.grad_bias = grad_bias
        return grad_input

    def step(self, step_size):
        self.kernel -= step_size * self.grad_weights
        self.bias -= step_size * self.grad_bias



class CLASS_AVG_POOLING:
    def __init__(self,IN,IH,IW,ON):
        # Initialize parameters
        self.IN = IN
        self.IH = IH
        self.IW = IW
        self.ON = ON
    # Forward pass 
    def forward(self, input):
        # print("\n\n\n-----------AVG POOLING-----------")
        output = np.zeros((batch_size,self.ON)) # 1x1xON
        for B in range(batch_size):
            for i in range(input.shape[0]):
                data = input[B,i].reshape((1,self.IH*self.IW))
                output[B,i] = np.average(data)
            # print("AVG POOLING:[i] ", output[i])
        # print("AVG POOLING all ", output)
        return output

    # Backward pass masks out same elements

    def backward(self, grad):
        # grad size 
        # Resize grad dimension (1 x 64) -> (64 x 25 x 5)
        input_grad = np.zeros((batch_size,self.IN, self.IH, self.IW))
        for B in range(batch_size):
            for i in range(grad.shape[1]):
                matrix = np.ones((self.IH, self.IW)) * (grad[B][i]/(self.IH*self.IW))
                input_grad[B,i] = matrix
        # input grad dimension  (64 x 25 x 5)
        # print("input_grad backward return:", input_grad.shape)
        return input_grad
    # No parameters so nothing to do during a gradient descent step

    def step(self, step_size):
        return


class CLASS_FULLY_CONNECTED:
    def __init__(self,IN,ON):
        # Initialize parameters
        self.IN = IN
        self.ON = ON

        self.KN = IN*ON

        self.kernel = np.random.randn(self.IN, self.ON)
        self.bias = np.random.rand(self.ON, 1)        #change to array shape 
    # Forward pass is max(0,input)

    def forward(self, input):
        self.input = input.reshape((batch_size,1, 64))
        # INPUT dimension (1 x 64)
        # INPUT T dimension (64 x 1)
        # KERNEL dimension (64 x 12)
        # OUTPUT dimension (12 x 1) --> Transpose here 
        return (np.dot(self.input, self.kernel) + np.transpose(self.bias)).reshape((batch_size, 12))

    # Backward pass masks out same elements
    def backward(self, grad):
        
        grad_bias = np.zeros(self.bias.shape)
        #   dL/d_input = (dL/d_output) * (d_output/d_input)
        #                          where (d_output/d_input) = wT
        
        # grad_input dimension 1 x 64
        # kernel dimension 64 x 12
        # kernel Transpose dimension 12 x 64
        # grad dimension 1 x 12
        grad_input = grad@np.transpose(self.kernel)
        # compute gradient w.r.t. weights and biases
        # dL/dW = (input)T * (dL/d_Z)
        # dL/dB = Sum of (dL/dZ)
        # grad_input dimension (1 x 64) -T-> (64 x 1)
        # grad dimension: (1 x 12)
        # print("FC self.input[]: ", self.input.shape)
        # print("FC grad[]: ", grad.shape)
        self.grad_weights = (np.transpose(self.input)@grad).reshape(64,12)
        
        # for B in range(batch_size):
        for M in range(self.ON):
            grad_bias[M] = np.average(grad)
            print("npnp.average(grad): ",np.average(grad))
        # self.grad_bias = np.sum(grad, axis=0)
        print("self.grad_weight: ", np.shape(self.grad_weights))
        self.grad_bias = grad_bias
        
        print("self.input: ", np.shape(self.input))
        print("grad: ", np.shape(grad))

        return grad_input

    def step(self, step_size):
        self.kernel -= step_size * self.grad_weights
        self.bias -= step_size * self.grad_bias


class CLASS_KWS_NETWORK:
    def __init__(self):
        # def __init__(          IN,IH,IW,PHu,PHd,PWl,PWr,KH,KW,ST,ON)
        self.layers = [CLASS_CONV(1,49,10,4,5,1,1,10,4,2,64)]
        self.layers.append(CLASS_ReLU())
        for i in range(number_of_hidden_layers):
            # print("hidden layer [%d] will goes through ", i)
            self.layers.append(CLASS_D_CONV(64,25,5,1,1,1,1,3,3,1,64))
            self.layers.append(CLASS_ReLU())
            self.layers.append(CLASS_P_CONV(64,25,5,1,1,64))
            self.layers.append(CLASS_ReLU())
        self.layers.append(CLASS_AVG_POOLING(64,25,5,64))
        self.layers.append(CLASS_FULLY_CONNECTED(64,12))

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
        # print("\nRelu passed grad[]: ", grad)
        # print("  Relu self mask[]: ", self.mask.shape)
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
        self.labels = labels.astype(int)

        return -np.mean(np.log(self.probs[np.arange(len(self.probs))[:, np.newaxis], self.labels]+0.00001))


    def backward(self):
        grad = self.probs
        grad[np.arange(len(self.probs))[:, np.newaxis], self.labels] -= 1

        # Equation #18 first value
        return grad.astype(np.float64)/len(self.probs)


if __name__ == "__main__":
    main()
