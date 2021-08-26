import matplotlib
import numpy as np
import os.path
import sys
import librosa  # for audio related library using
from os import listdir
from os.path import isfile, join
from os import walk

# make a function as call from above
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
    # y, sr = librosa.load(librosa.ex('trumpet'))
    audio_data, sr = librosa.load(wav_file, sr=sr) # , offset=0.04, duration=1.0)
    print('---audio_sr:',sr)
    print('---audio_data:',audio_data.shape)
    audio_np = np.array(audio_data, np.float32)
    print('audio_np:', audio_np.shape)

    mfcc_librosa = librosa.feature.mfcc(y=audio_data, sr=sr,
                                        win_length=win_length, hop_length=hop_length,
                                        center=False, # it will be start FFT from begins at y[t* hop_length]
                                        n_fft=n_fft,
                                        n_mfcc=n_mfcc, n_mels=n_mels,
                                        fmin=fmin, fmax=fmax, htk=False
                                    )

    print('mfcc_librosa',mfcc_librosa.shape)


# get wav file from system OS
files = [f for f in os.listdir('/Users/hongyielsuh/Documents/GitHub/KWSproject_python') if os.path.isfile(f)]
# get files directory
for f in files:
    # print(f)
    wav_constructor = os.path.splitext(f)[1]
    # make_matrix(wav_file)
    print(wav_constructor)
    # do something in here
    if wav_constructor == ".wav":
        make_matrix(f)

# packaging the file matrix to 49 x 10 x 1 for original statement

# distinguish the answer through the file direction that from the system input

# export format in csv file (IMPORTANT:: format should to in line 490 x (number of files))


