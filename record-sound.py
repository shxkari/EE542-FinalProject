import sounddevice as sd
from scipy.io.wavfile import write
import wave
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import librosa
import numpy as np
import time
fs = 22050  # Sample rate
seconds = 2  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
print("RECORDING AUDIO")
print(myrecording.dtype)
print(myrecording.shape)
print(myrecording)
sd.wait()  # Wait until recording is finished
print("DONE")
write('output.wav', fs, myrecording)  # Save as WAV file 

file_name = "output.wav"
myrecording, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
print(myrecording)

# write('output.wav', fs, myrecording)  # Save as WAV file 

# unpickling the pickles
loaded_model = pickle.load(open('CNN_model.pickle', 'rb'))
le = pickle.load(open('LE.pickle', 'rb'))

num_rows = 40
num_columns = 174
num_channels = 1
max_pad_len = 174

def extract_features(data):
    mfccs = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=num_rows, hop_length=512)
    pad_width = max_pad_len - mfccs.shape[1]
    if(pad_width < 0):
        print(file_name)
        mfccs = mfccs[:,:pad_width].copy()
    else:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

def print_prediction(data):
    prediction_feature = extract_features(data) 
    #print(prediction_feature)
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)
    #print(prediction_feature)
    time_now = time.time()
    predicted_vector = loaded_model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 
    
    predicted_proba_vector = loaded_model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )
    print("Prediction took {} seconds".format(time.time() - time_now))
print_prediction(myrecording)




