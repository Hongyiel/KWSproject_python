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
    
    # if key_word_constructor in "wav_data_down 10"
    # if key_word_constructor in "wav_data_go"  1:
    # if key_word_constructor in "wav_data_left" 2:
    # if key_word_constructor in "wav_data_no" 3:
    # if key_word_constructor in "wav_data_off" 4:
    # if key_word_constructor in "wav_data_on" 5:
    # if key_word_constructor in "wav_data_right" 6:
    # if key_word_constructor in "wav_data_stop 7" :
    # if key_word_constructor in "wav_data_up 8":
    # if key_word_constructor in "wav_data_yes 9":
    
    # if +silence 11
    # if unknown 12


