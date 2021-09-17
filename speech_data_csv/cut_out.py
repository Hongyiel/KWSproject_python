# this file is for devide data as 200
import pandas as pd
import numpy as np
import random
import sklearn

import csv
import os

files = [f for f in os.listdir('/Users/hongyielsuh/Documents/GitHub/KWSproject_python/speech_data_csv') if os.path.isfile(f)]
frames = []
su_frames = []
print(files)
 

i = 0

for f in files:
    
    key_word_constructor = os.path.splitext(f)[1]
    file_name_constructor = os.path.splitext(f)[0]

    if key_word_constructor == ".csv":
        if file_name_constructor == "wav_data_silence_" or file_name_constructor == "wav_data_unknown_":
            # print("file_name_constructor: ",file_name_constructor)
            # print("Gotcha!")
            data = pd.read_csv(f,header=None)
            su_frames.append(data)
                
            # data = pd.read_csv(f,header=None)
            # frames.append(data)
        else:
            # print("file_name_constructor: ",file_name_constructor)
            # print("THE BIG BOY")

            data = pd.read_csv(f,header=None)
            frames.append(data)
            
            
            
            # print("file_name_constructor: ",file_name_constructor)
            # print("Gotcha!")
            # data = pd.read_csv(f,header=None)
            # su_frames.append(data)

data_set = []

for sp in frames:
    df_shuffled=sklearn.utils.shuffle(sp)

    # sp = np.random.permutation(sp)
    # print(sp)
    i = i +1
    data_set.append(df_shuffled[0:350])
    print(i)
    # print(sp)

# print(data_set)
print(data_set[0])
# print(su_frames)
print("Second line")
for su in su_frames:
    df_shuffled=sklearn.utils.shuffle(su)

    i = i + 1
    data_set.append(df_shuffled[0:200])
    print(i)
print(data_set[0])

# print("check point")
# # print(data_set)
# # print(frames[1][1])

df = pd.concat(data_set)
# print(df)
print(np.shape(df))
df.to_csv("data_temp.csv")

# print(np.shape("data_temp.csv"))