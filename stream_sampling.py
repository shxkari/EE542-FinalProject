import sounddevice as sd
from scipy.io.wavfile import write
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import librosa
import numpy as np
import time
import threading
from queue import Queue

SAMPLING_RATE = 22000
SECONDS_SAMPLED = 4
OVERLAP_TIME = 1

def extract_features(data):
    mfccs = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=40)
    print(mfccs.shape)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

def print_prediction(predict_done, e,queue):
    while(1):
        data = queue.get()
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


def windowing(e_mine, e_other):
    while(1):
        flagready = 0
        data = []
        #start sampling for how many seconds we want
        for i in range (SAMPLING_RATE * SECONDS_SAMPLED):
            data_buffer.append (***fill out****)
            if (i == SAMPLING_RATE * (SECONDS_SAMPLED - OVERLAP_TIME)):
                e_mine.set()
        myrecording = np.array(data)
        write('output.wav', fs, myrecording)  # Save as WAV file 
        file_name = "output.wav"
        myrecording, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 

        print_prediction(myrecording)
        e_other.wait()
        e_other.clear()
  
def parse_line(in_line):
    # array of strings split on delimiter
    # convert to an array of floats 
    data = in_line.split(',')
    data = np.array(data)
    print(data)
    data = data.astype(np.int16)
    print(data)
    return data


if __name__ == "__main__": 
    # unpickling the pickles
    loaded_model = pickle.load(open('CNN_model.pickle', 'rb'))
    le = pickle.load(open('LE.pickle', 'rb'))

    num_rows = 40
    num_columns = 174
    num_channels = 1
    max_pad_len = 174
    


    e1 = threading.Event()
    e2 = threading.Event()
    t1 = threading.Thread(target=windowing, args=(e1,e2,))
    t2 = threading.Thread(target=windowing, args=(e2,e1,))

'''
#myrecording = np.frombuffer(data, dtype=np.uint16)
myrecording = np.array(data)
for i in myrecording:
    print(i)
myrecording =np.asfarray(myrecording,float)

write('output.wav', fs, myrecording)  # Save as WAV file 


file_name = "output.wav"
myrecording, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
print(myrecording)
'''
# write('output.wav', fs, myrecording)  # Save as WAV file 



#print_prediction(myrecording)



