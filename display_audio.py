import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import sys

import librosa.display

max_pad_len = 174
mfcc_count = 60

def extract_features(data, fs):
    mfccs = librosa.feature.mfcc(y=data, lifter=mfcc_count, sr=sample_rate, n_mfcc=mfcc_count)
    pad_width = max_pad_len - mfccs.shape[1]
    if(pad_width < 0):
        mfccs = mfccs[:,:pad_width].copy()
    else:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

audio = AudioSegment.from_wav(sys.argv[1], )
frame_count = audio.frame_count()
frame_rate = audio.frame_rate
samples = audio.get_array_of_samples()

print("Number of channels: " + str(audio.channels))
print("Frame count: " + str(frame_count))
print("Frame rate: " + str(frame_rate))
print("Sample width: " + str(audio.sample_width))
print("Number of samples: " + str(len(samples)))

"""***********************FULL SAMPLE PLOT**************************"""
period = 1/frame_rate                     #the period of each sample
duration = frame_count/frame_rate         #length of full audio in seconds
time = np.arange(0, duration, period)     #generate a array of time values from 0 to [duraction] with step of [period]

#TODO: Plot the full sample as a subplot
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
# plt.plot(time, samples)
plt.title('Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')

myrecording, sample_rate = librosa.load(sys.argv[1], res_type='kaiser_fast') 
feat = extract_features(myrecording, frame_rate)
plt.subplot(1,2,2)
librosa.display.specshow(feat, x_axis='time', )
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()