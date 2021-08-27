import matplotlib
import numpy as np
import os.path
import sys
import librosa  # for audio related library using
from os import listdir
from os.path import isfile, join
from os import walk
import csv
# import pandas as pd


# make a audio file to matrix 49 x 10 x 1
def make_matrix(wav_file):
    # wav_file = "go_nohash_0.wav"
    n_mfcc = 49
    n_mels = 10
    n_fft = 640 ## 512 
    win_length = 640 # 160
    hop_length = 320 # 160
    fmin = 20
    fmax = 4000
    sr = 16000
    # how to call data from librosa
    audio_data, sr = librosa.load(wav_file, sr=sr) # , offset=0.04, duration=1.0)
    audio_np = np.array(audio_data, np.float32)

    mfcc_librosa = librosa.feature.mfcc(y=audio_data, sr=sr,
                                        win_length=win_length, hop_length=hop_length,
                                        center=False, # it will be start FFT from begins at y[t* hop_length]
                                        n_fft=n_fft,
                                        n_mfcc=n_mfcc, n_mels=n_mels,
                                        fmin=fmin, fmax=fmax, htk=False
                                    )
    return mfcc_librosa

# merge_matrix as 490 x 1 x 1 format
def merge_matrix(wav_matrix):
    new_matrix = []
    for k in wav_matrix:
        new_matrix.extend(k)
        # make a format as 490 x 1 x 1
        # export format in csv file (IMPORTANT:: format should to in line 490 x (number of files))
    return new_matrix

# get wav file from system OS
files = [f for f in os.listdir('/Users/hongyielsuh/Documents/GitHub/KWSproject_python') if os.path.isfile(f)]
# get files directory
final_matrix = []
for f in files:
    wav_constructor = os.path.splitext(f)[1]
    if wav_constructor == ".wav":
        print("wav_constructor: ", wav_constructor)
        print("file name: ", f)
        wav_matrix = make_matrix(f)
        # packaging the file matrix to 49 x 10 x 1 for original statement (above def will process it)
        # distinguish the answer through the file direction that from the system input
        final_matrix.append(merge_matrix(wav_matrix))
        print(np.shape(final_matrix))

np.savetxt('wav_data.csv', final_matrix, delimiter=',') 
# set matrix for import main python file