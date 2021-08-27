import matplotlib
import numpy as np
from pathlib import Path

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
    print("this is wav_=file: " ,wav_file)
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
    print("error found lineerror found lineerror found lineerror found lineerror found line")
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
files = [f for f in os.listdir('/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/go') if os.path.isfile(f)]

file_list = []
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02
for p in Path('/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/go').glob('*.wav'):
    # print(f"{p.name}")
    # print("###")
    # print(f"{p.name}")
    file_list.append(f"{p.name}")

# print(file_list)
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/_silence_
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/_unknown_
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/down
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/go
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/left
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/no
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/off
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/on
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/right
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/stop
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/up
#/Users/hongyielsuh/Desktop/speech_dataset/speech_commands_test_set_v0.02/yes


# get folder name
# get file name

# if the files are including folder that name called "go" then attatch the number "1"
# if the files are including folder that name called "left" then attatch the number "2"
# ...



# get files directory
final_matrix = []
# print("all file: ", file_list)
for f in file_list:
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