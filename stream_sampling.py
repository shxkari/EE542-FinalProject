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

SAMPLING_RATE = 22000
SECONDS_SAMPLED = 4
OVERLAP_TIME = 1

# unpickling the pickles
# raw_model = open('CNN_model.pickle', 'rb')
# raw_le = open('LE.pickle', 'rb')
# loaded_model = pickle.load(open('CNN_model.pickle', 'rb'))
# le = pickle.load(open('LE.pickle', 'rb'))

num_rows = 40
num_columns = 174
num_channels = 1
max_pad_len = 174

# myrecording_1
# myrecording_2

def extract_features(data):
    mfccs = librosa.feature.mfcc(y=data, sr=SAMPLING_RATE, n_mfcc=40)
    print(mfccs.shape)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

def print_prediction(data, loaded_model, le):

    prediction_feature = extract_features(data) 
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)
    print(prediction_feature)
    predicted_vector = loaded_model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 
    
    predicted_proba_vector = loaded_model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )


def windowing(id,e_mine, e_other):
    loaded_model = pickle.load(open('CNN_model.pickle', 'rb'))
    le = pickle.load(open('LE.pickle', 'rb'))
    e_mine.wait()
    while(1):
        flagready = 0
        data = []
        #start sampling for how many seconds we want
        for i in range (SAMPLING_RATE * SECONDS_SAMPLED):
            data.append (0)
            if (i == SAMPLING_RATE * (SECONDS_SAMPLED - OVERLAP_TIME)):
                e_mine.set()
        myrecording = np.array(data, dtype=np.int16)
        myrecording = myrecording.reshape((myrecording.shape[0],1))
        print(myrecording.shape)
        # myrecording =np.asfarray(myrecording,float)
        write('output.wav', SAMPLING_RATE, myrecording, )  # Save as WAV file 
        file_name = "output.wav"
        myrecording, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        print(myrecording)
        print_prediction(myrecording, loaded_model, le)
        e_other.wait()
        e_other.clear()
  
'''  
def parse_line(in_line): 
    data = in_line.split(',')
    data = np.array(data)
    print(data)
    data = data.astype(np.int16)
    print(data)
    return data
'''

if __name__ == "__main__": 

    e1 = threading.Event()
    e2 = threading.Event()
    t1_ID = 1
    t2_ID = 2
    t1 = threading.Thread(target=windowing, args=(t1_ID,e1,e2,))
    t2 = threading.Thread(target=windowing, args=(t2_ID,e2,e1,))
    e1.set()
    t1.start()
    t2.start()



