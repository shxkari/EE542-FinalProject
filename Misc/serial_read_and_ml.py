import sounddevice as sd
from scipy.io.wavfile import write
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import librosa
import numpy as np
import serial 
from serial import Serial
import time

data = []
fs = 10000  # Sample rate
seconds = 0.8  # Duration of recording
ser = serial.Serial('/dev/tty.usbmodem1441202')
ser.flushInput()

print("Hello World!")

while(1):
    line = ser.readline()
    if(len(line) == 0):
        continue
    decoded_line = str(line, 'utf-8')
    if('Time' in decoded_line):
        break

time_str = decoded_line
times = time_str.split(":")
time = times[1]
print(time)
sr =  8000 * 10**6 / int(time);
#print("sampling rate: ")
print(sr)

while(1): 
    line = ser.readline()
    decoded_line = str(line, 'ascii')
    if('start' in decoded_line):
        break

print("starting to send from xdot to laptop")


while (1):
    line = ser.readline()
    decoded_line = str(line, 'ascii')
    if("end" in decoded_line):
        break
    data.append(decoded_line); 
    print(decoded_line)



myrecording = np.array(data)

print("done reading data")

for i in myrecording:
    print(i)
myrecording =np.asfarray(myrecording,np.int16)
write('output.wav', int(sr), myrecording)  # Save as WAV file 


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
    mfccs = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=40)
    print(mfccs.shape)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

def print_prediction(data):
    prediction_feature = extract_features(data) 
    #print(prediction_feature)
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)
    #print(prediction_feature)
    predicted_vector = loaded_model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 
    
    predicted_proba_vector = loaded_model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )

print_prediction(myrecording)

ser.close()


