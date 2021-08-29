# this file for merge dataset from speech data

import csv
import glob
import os
import numpy as np

# bring all csv file
# file file name including "go" then add 1 column and replace it


files = [f for f in os.listdir('/Users/hongyielsuh/Documents/GitHub/KWSproject_python/speech_data_csv') if os.path.isfile(f)]

for f in files:
    key_word_constructor = os.path.splitext(f)[0]
    print(key_word_constructor)
    
    if key_word_constructor in "wav_data_yes":
        print("yes!")
    if key_word_constructor in "wav_data_go":
        with open(f, 'r') as csvoutput:
            print(np.shape(f))

    # if key_word_constructor in "wav_data_left":
    # if key_word_constructor in "wav_data_down":
    # if key_word_constructor in "wav_data_up":

    # if key_word_constructor in "wav_data_no":
    # if key_word_constructor in "wav_data_off":

