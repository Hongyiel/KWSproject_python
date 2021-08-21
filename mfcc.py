import matplotlib
import numpy as np
import os.path
import sys
import librosa  # for audio related library using

wav_file = "go_nohash_0.wav"
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