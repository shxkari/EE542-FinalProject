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
SECONDS_SAMPLED = 1

def windowing(data_buffer, e, queue):
    while(1):
        data_buffer.clear()
        for i in range (SAMPLING_RATE * SECONDS_SAMPLED):
            data_buffer.append (***fill out****)
            if (i == 2000):
                flag = 1
        myrecording = np.array(data)
        write('output.wav', fs, myrecording)  # Save as WAV file 

        file_name = "output.wav"
        myrecording, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        queue.put(myrecording)
        
        event_is_set = e.wait()
        e.clear()
  
def parse_line(in_line):
    # array of strings split on delimiter
    # convert to an array of floats 
    data = in_line.split(',')
    data = np.array(data)
    print(data)
    data = data.astype(np.int16)
    print(data)
    return data

def extract_features(data):
    mfccs = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=40)
    print(mfccs.shape)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

def print_prediction(queue):
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

if __name__ == "__main__": 
    # unpickling the pickles
    loaded_model = pickle.load(open('CNN_model.pickle', 'rb'))
    le = pickle.load(open('LE.pickle', 'rb'))

    num_rows = 40
    num_columns = 174
    num_channels = 1
    max_pad_len = 174
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []

    flag1 = 0
    flag2 = 0

    my_queue = Queue.Queue()
    e1 = threading.Event()
    e2 = threading.Event()
    t1 = threading.Thread(target=windowing, args=(data1,flag1,my_queue,e1))
    t2 = threading.Thread(target=windowing, args=(data2,flag2,my_queue,e2))
    #t3 = threading.Thread(target=windowing, args=(data3,))
    #t4 = threading.Thread(target=windowing, args=(data4,))
    #t5 = threading.Thread(target=windowing, args=(data5,))
    tpredict1 = threading.Thread(target=print_prediction, args=(my_queue))
    tpredict2 = threading.Thread(target=print_prediction, args=(my_queue))

    t1.start()
    if(flag1):
        t2.start()
        flag1 = 0
    if(flag2):
        t3.start()
        flag2 = 0

    t1.join()
    tpredict1.start()
    t2.join()
    tpredict2.start()

    tpredict1.join()
    e1.set()
    tpredict2.join()
    e2.set()

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


#print_prediction(myrecording)



