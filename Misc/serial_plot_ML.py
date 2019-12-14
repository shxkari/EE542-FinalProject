# serial stuff

import serial
import numpy as np
from matplotlib import pyplot as plot
import csv
import sys
import pickle
from scipy.io.wavfile import write
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import librosa

def parse_line(in_line):
	# array of strings split on delimiter
	# convert to an array of floats 
	data = in_line.split(",")
	data = np.array(data)
	print(data)
	data = data.astype(np.int16)
	print(data)
	return data
'''
ser = serial.Serial("/dev/tty.usbmodem1413202")

line = ser.readline()

print(line)'''
arr = []

# unpickling the pickles
loaded_model = pickle.load(open('CNN_model.pickle', 'rb'))
le = pickle.load(open('LE.pickle', 'rb'))

num_rows = 40
num_columns = 174
num_channels = 1

with open('raw_adc.txt', 'r') as fp:
	line = fp.readline()
	while(len(line) > 0):
		arr.append(parse_line(line))
		line = fp.readline()


write('output.wav', 8000, arr)  # Save as WAV file 

