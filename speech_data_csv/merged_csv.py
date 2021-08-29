# this file for merge dataset from speech data

import csv
import glob
import os
import numpy as np
import pandas as pd


# bring all csv file
# file file name including "go" then add 1 column and replace it

path = '/Users/hongyielsuh/Documents/GitHub/KWSproject_python/speech_data_csv'
files = [f for f in os.listdir('/Users/hongyielsuh/Documents/GitHub/KWSproject_python/speech_data_csv') if os.path.isfile(f)]
frames= []

for f in files:
    
    key_word_constructor = os.path.splitext(f)[1]
    if key_word_constructor == ".csv":
        data = pd.read_csv(f,header=None)
        
        frames.append(data)
    
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
    
df = pd.concat(frames)
print(df)

df.to_csv("data_set.csv")
# merging the files
# joined_files = os.path.join(path, "wav_data_*.csv")
# # A list of all joined files is returned
# joined_list = glob.glob(joined_files)
# # Finally, the files are joined

# df = pd.concat((pd.read_csv(file).assign(filename = file) for file in joined_list), ignore_index=True)
# print(df)
# print("process finished...")


# interesting_files = glob.glob("/Users/hongyielsuh/Documents/GitHub/KWSproject_python/speech_data_csv/*.csv") 
# df = pd.concat((pd.read_csv(f, header = None) for f in interesting_files))
# # df.to_csv("output.csv")

# print(df)

# data1 = pd.read_csv("wav_data_up.csv",header=None)
# data2 = pd.read_csv("wav_data_on.csv",header=None)


# frames = [data1, data2]
# df = pd.concat(frames)

# print(df)