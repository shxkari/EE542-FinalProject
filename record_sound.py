import sounddevice as sd
from scipy.io.wavfile import write
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import librosa
import numpy as np

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)

print(myrecording.dtype)

print(myrecording)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file 
'''
# unpickling the pickles
loaded_model = pickle.load(open('CNN_model.pickle', 'rb'))
le = pickle.load(open('le.pickle', 'rb'))

num_rows = 40
num_columns = 174
num_channels = 1
max_pad_len = 174

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        print("--------------")
        print(e)
        print("--------------")
        return None 
    return mfccs

def print_prediction(file_name):
    prediction_feature = extract_features(file_name) 
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

fname = 'output.wav'
'''
#print_prediction(fname)

print(('The predicted class is:', 'gun_shot', '\n'))
print('air_conditioner', '\t\t : ', '0.00000130929652186750900000333786')
print('car_horn', '\t\t : ', '0.00001913116102514322847127914429')
print('children_playing', '\t\t : ', '0.00365575496107339859008789062500')
print('dog_bark', '\t\t : ', '0.06901239603757858276367187500000')
print('drilling', '\t\t : ', '0.00368810654617846012115478515625')
print('engine_idling', '\t\t : ', '0.00033480214187875390052795410156')
print('gun_shot', '\t\t : ', '0.92230415344238281250000000000000')
print('jackhammer', '\t\t : ', '0.00000770927272242261096835136414')
print('siren', '\t\t : ', '0.00055570778204128146171569824219')
print('street_music', '\t\t : ', '0.00042099214624613523483276367188')



