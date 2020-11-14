#!/usr/bin/env python
# coding: utf-8

# In[2]:



get_ipython().system('pip install tensorflow ')


# In[3]:


get_ipython().system('pip install keras')


# In[13]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 

sns.set()


# In[14]:


get_ipython().system('pip install kaggle')


# In[15]:


#read csv file
dataset = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')
dataset.rename(columns={'0':'label'}, inplace=True)

# x vector
X = dataset.drop('label',axis = 1)

# label
# labels are number now --> need to convert to letters later
y = dataset['label']


# In[16]:


# view dataset
print(dataset)


# In[17]:


# size of X
print("shape:",X.shape)
# num features
print("culoms count:",len(X.iloc[1]))


# view first 5 rows
X.head()


# In[18]:


from sklearn.utils import shuffle

X_shuffle = shuffle(X)

plt.figure(figsize = (12,10))
row, colums = 4, 4

# view the first 16 images
for i in range(16):  
    plt.subplot(colums, row, i+1)
    plt.imshow(X_shuffle.iloc[i].values.reshape(28,28),interpolation='nearest', cmap='Greys')
plt.show()


# In[19]:


print("Amount of each labels")

# Change label to letters
alphabets_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 
dataset_alphabets = dataset.copy()
dataset['label'] = dataset['label'].map(alphabets_mapper)

label_size = dataset.groupby('label').size()
label_size.plot.barh(figsize=(10,10))
plt.show()

#print("We have very low observations for I and F ")
#print("I count:", label_size['I'])
#print("F count:", label_size['F'])


# In[20]:


# splite the data
X_train, X_test, y_train, y_test = train_test_split(X,y)

# scale data
standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train)

X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)


# In[21]:


print("Data after scaler")
X_shuffle = shuffle(X_train)

plt.figure(figsize = (12,10))
row, colums = 4, 4
for i in range(16):  
    plt.subplot(colums, row, i+1)
    plt.imshow(X_shuffle[i].reshape(28,28),interpolation='nearest', cmap='Greys')
plt.show()


# In[22]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[23]:


# build the CNN
cls = Sequential()
cls.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
cls.add(MaxPooling2D(pool_size=(2, 2)))
cls.add(Dropout(0.3))
cls.add(Flatten())
cls.add(Dense(128, activation='relu'))
cls.add(Dense(len(y.unique()), activation='softmax'))

cls.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = cls.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=18, batch_size=200, verbose=2)

scores = cls.evaluate(X_test,y_test, verbose=0)
print("CNN Score:",scores[1])


# In[24]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[32]:


cm=confusion_matrix(y_test.argmax(axis=1),cls.predict(X_test).argmax(axis=1))
#print(cm)
df_cm = pd.DataFrame(cm, range(26),
                  range(26))
plt.figure(figsize = (20,15))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size


# In[26]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install -U coremltools')
import coremltools


# In[27]:


get_ipython().system('pip install tensorflowjs ')


# In[28]:


get_ipython().system('mkdir model')
get_ipython().system('tensorflowjs_converter --input_format keras my_model.h5 model/')


# In[29]:


cls.save('my_model.h5')


# In[30]:


get_ipython().system('zip -r model.zip model ')


# In[33]:



output_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#core_ml=coremltools.converters.keras.convert('my_model.h5', input_names=['image'], 
                                             #output_names=['output'], 
                                             #class_labels=output_labels,
                                             #image_scale=1/255.0, is_bgr = False, 
                                             #image_input_names = "image")
#core_ml.save('coreml_model.mlmodel')

cls.save('/Users/yichenwu/Desktop/model')


# In[34]:


cls.save_weights('/Users/yichenwu/Desktop/model')
cls.load_weights('/Users/yichenwu/Desktop/model')


# In[ ]:




