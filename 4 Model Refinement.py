#!/usr/bin/env python
# coding: utf-8

# # Classifying Urban sounds using Deep Learning
# 
# ## 4 Model Refinement 

# ### Load Preprocessed data 

# In[1]:


# retrieve the preprocessed data from previous notebook

get_ipython().magic(u'store -r x_train')
get_ipython().magic(u'store -r x_test')
get_ipython().magic(u'store -r y_train')
get_ipython().magic(u'store -r y_test')
get_ipython().magic(u'store -r yy')
get_ipython().magic(u'store -r le')


# #### Model refinement
# 
# In our inital attempt, we were able to achieve a Classification Accuracy score of: 
# 
# * Training data Accuracy:  92.3% 
# * Testing data Accuracy:  87% 
# 
# We will now see if we can improve upon that score using a Convolutional Neural Network (CNN). 

# #### Feature Extraction refinement 
# 
# In the prevous feature extraction stage, the MFCC vectors would vary in size for the different audio files (depending on the samples duration). 
# 
# However, CNNs require a fixed size for all inputs. To overcome this we will zero pad the output vectors to make them all the same size. 

# In[2]:


import numpy as np
max_pad_len = 174

def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs


# In[3]:


# Load various imports 
import pandas as pd
import os
import librosa

# Set the path to the full UrbanSound dataset 
fulldatasetpath = '/Users/candre/Documents/UrbanSound8K/audio'

metadata = pd.read_csv('../UrbanSound Dataset sample/metadata/UrbanSound8K.csv')

features = []

# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    
    class_label = row["class_name"]
    data = extract_features(file_name)
    
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files') 


# In[4]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 

# split the dataset 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)


# ### Convolutional Neural Network (CNN) model architecture 
# 
# 
# We will modify our model to be a Convolutional Neural Network (CNN) again using Keras and a Tensorflow backend. 
# 
# Again we will use a `sequential` model, starting with a simple model architecture, consisting of four `Conv2D` convolution layers, with our final output layer being a `dense` layer. 
# 
# The convolution layers are designed for feature detection. It works by sliding a filter window over the input and performing a matrix multiplication and storing the result in a feature map. This operation is known as a convolution. 
# 
# 
# The `filter` parameter specifies the number of nodes in each layer. Each layer will increase in size from 16, 32, 64 to 128, while the `kernel_size` parameter specifies the size of the kernel window which in this case is 2 resulting in a 2x2 filter matrix. 
# 
# The first layer will receive the input shape of (40, 174, 1) where 40 is the number of MFCC's 174 is the number of frames taking padding into account and the 1 signifying that the audio is mono. 
# 
# The activation function we will be using for our convolutional layers is `ReLU` which is the same as our previous model. We will use a smaller `Dropout` value of 20% on our convolutional layers. 
# 
# Each convolutional layer has an associated pooling layer of `MaxPooling2D` type with the final convolutional layer having a `GlobalAveragePooling2D` type. The pooling layer is do reduce the dimensionality of the model (by reducing the parameters and subsquent computation requirements) which serves to shorten the training time and reduce overfitting. The Max Pooling type takes the maximum size for each window and the Global Average Pooling type takes the average which is suitable for feeding into our `dense` output layer.  
# 
# Our output layer will have 10 nodes (num_labels) which matches the number of possible classifications. The activation is for our output layer is `softmax`. Softmax makes the output sum up to 1 so the output can be interpreted as probabilities. The model will then make its prediction based on which option has the highest probability.

# In[5]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

num_rows = 40
num_columns = 174
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)


print(x_train.size)


num_labels = yy.shape[1]
filter_size = 2

# Construct model 
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax')) 


# ### Compiling the model 
# 
# For compiling our model, we will use the same three parameters as the previous model: 

# In[6]:


# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 


# In[7]:


# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)


# ### Training 
# 
# Here we will train the model. As training a CNN can take a sigificant amount of time, we will start with a low number of epochs and a low batch size. If we can see from the output that the model is converging, we will increase both numbers.  

# In[8]:


from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

#num_epochs = 12
#num_batch_size = 128

num_epochs = 50
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# ### Test the model 
# 
# Here we will review the accuracy of the model on both the training and test data sets. 

# In[9]:


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])


# The Training and Testing accuracy scores are both high and an increase on our initial model. Training accuracy has increased by ~6% and Testing accuracy has increased by ~4%. 
# 
# There is a marginal increase in the difference between the Training and Test scores (~6% compared to ~5% previously) though the difference remains low so the model has not suffered from overfitting. 

# ### Predictions  
# 
# Here we will modify our previous method for testing the models predictions on a specified audio .wav file. 

# In[10]:


def print_prediction(file_name):
    prediction_feature = extract_features(file_name) 
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 
    
    predicted_proba_vector = model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )


# ### Validation 
# 
# #### Test with sample data 
# 
# As before we will verify the predictions using a subsection of the sample audio files we explored in the first notebook. We expect the bulk of these to be classified correctly. 

# In[11]:


# Class: Air Conditioner

filename = '../UrbanSound Dataset sample/audio/100852-0-0-0.wav' 
print_prediction(filename) 


# In[12]:


# Class: Drilling

filename = '../UrbanSound Dataset sample/audio/103199-4-0-0.wav'
print_prediction(filename) 


# In[13]:


# Class: Street music 

filename = '../UrbanSound Dataset sample/audio/101848-9-0-0.wav'
print_prediction(filename) 


# In[14]:


# Class: Car Horn 

filename = '../UrbanSound Dataset sample/audio/100648-1-0-0.wav'
print_prediction(filename) 


# #### Observations 
# 
# We can see that the model performs well. 
# 
# Interestingly, car horn was again incorrectly classifed but this time as drilling - though the per class confidence shows it was a close decision between car horn with 26% confidence and drilling at 34% confidence.  

# ### Other audio
# 
# Again we will further validate our model using a sample of various copyright free sounds that we not part of either our test or training data. 

# In[15]:


filename = '../Evaluation audio/dog_bark_1.wav'
print_prediction(filename) 


# In[16]:


filename = '../Evaluation audio/drilling_1.wav'

print_prediction(filename) 


# In[17]:


filename = '../Evaluation audio/gun_shot_1.wav'

print_prediction(filename) 


# In[20]:


import pickle
pickle_out = open("CNN_model.pickle","wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# #### Observations 
# 
# The performance of our final model is very good and has generalised well, seeming to predict well when tested against new audio data. 

# In[21]:


print(model)


# In[22]:


pickle_out2 = open("le.pickle","wb")
pickle.dump(le, pickle_out2)
pickle_out.close()


# In[ ]:




