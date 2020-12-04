#!/usr/bin/env python
# coding: utf-8

# In[2]:



get_ipython().system('pip install tensorflow ')


# In[3]:


get_ipython().system('pip install keras')


# In[25]:


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


# In[75]:



# x vector
X = dataset.drop('label',axis = 1)

# label
# labels are number now --> need to convert to letters later
y_label = dataset['label']

print(dataset.head())


# In[70]:


# view dataset
print(dataset.head())
print('y')
print(y_label)


# In[71]:


# size of X
print("shape:",X.shape)
# num features
print("culoms count:",len(X.iloc[1]))


# view first 5 rows
X.head()

print(dataset.head())


# In[ ]:





# In[34]:


from sklearn.utils import shuffle

X_shuffle = shuffle(X)

plt.figure(figsize = (12,10))
row, colums = 4, 4

# view the first 16 images
for i in range(16):  
    plt.subplot(colums, row, i+1)
    # using grey scale
    plt.imshow(X_shuffle.iloc[i].values.reshape(28,28),interpolation='nearest', cmap='Greys')
plt.show()


# In[53]:


import doctest
# to braille
def ostring_to_raisedpos(s):
    ''' (str) -> str
    Convert a braille letter represented by '##\n##\n##' o-string format
    to raised position format. Provided to students. Do not edit this function.

    Braille cell dot position numbers:
    1 .. 4
    2 .. 5
    3 .. 6
    7 .. 8 (optional)

    >>> ostring_to_raisedpos('..\\n..\\n..')
    ''
    >>> ostring_to_raisedpos('oo\\noo\\noo')
    '142536'
    >>> ostring_to_raisedpos('o.\\noo\\n..')
    '125'
    >>> ostring_to_raisedpos('o.\\noo\\n..\\n.o')
    '1258'
    '''
    res = ''
    inds = [1, 4, 2, 5, 3, 6, 7, 8]
    s = s.replace('\n', '')
    for i, c in enumerate(s):
        if c == 'o':
            res += str(inds[i])
    return res 


def raisedpos_to_binary(s):
    ''' (str) -> str
    Convert a string representing a braille character in raised-position
    representation  into the binary representation.
    TODO: For students to complete.

    >>> raisedpos_to_binary('')
    '00000000'
    >>> raisedpos_to_binary('142536')
    '11111100'
    >>> raisedpos_to_binary('14253678')
    '11111111'
    >>> raisedpos_to_binary('123')
    '11100000'
    >>> raisedpos_to_binary('125')
    '11001000'
    '''
    #define the templet we are going to subsitute on as standard with 8 letters
    standard = 'abcdefgh'

    #this is my overall method
    #1)find the correct position of letters on standard which need to be subsituted
    #2)subsitute it with 1
    #3)renew standard each time
    #4)after subsituting all the 1s, subsitute all the remaining letters with 0s
    

    #get access to all numbers in s
    for i in range(len(s)):
        #each charcters in s represent positions of 1 in cell
        position_of_1_in_cell = s[i] #this is a str
        #find the index on the standard where we need to substitute
        position_in_standard_need_to_be_changed = int(position_of_1_in_cell)-1
        #subsitute the element in standard with the correct index with 1
        standard = standard.replace(standard[position_in_standard_need_to_be_changed], '1')

    #get access to all characters in standard now   
    for j in range(len(standard)): #[0,8)
        #find out the position in standard which hasn't been substituted by 1
        if (standard[j] != '1'):
            #subsitute thoes all with 0s
            standard = standard.replace(standard[j],'0')
    
    return standard


def binary_to_hex(s):
    '''(str) -> str
    Convert a Braille character represented by an 8-bit binary string
    to a string representing a hexadecimal number.

    TODO: For students to complete.

    The first braille letter has the hex value 2800. Every letter
    therafter comes after it.

    To get the hex number for a braille letter based on binary representation:
    1. reverse the string
    2. convert it from binary to hex
    3. add 2800 (in base 16)

    >>> binary_to_hex('00000000')
    '2800'
    >>> binary_to_hex('11111100')
    '283f'
    >>> binary_to_hex('11111111')
    '28ff'
    >>> binary_to_hex('11001000')
    '2813'
    '''
    #reverse the string
    reverse_str = s[::-1]
    #convert the reverse_str frm binary to decimal 
    dec_str = int(reverse_str, 2)
    # convert 2800 in base 16 to decimal numebr
    dec_2800 = int('2800',16)
    #add two decimal numbers together
    summation = dec_2800 + dec_str
    #convert them into hexadecimal number
    #summation_in_hex is a str starts with 0x!
    summation_in_hex = hex(summation)
    #slice the str, cut off the first two character
    what_we_want = summation_in_hex[2:6]
    
    return what_we_want


def hex_to_unicode(n):
    '''(str) -> str
    Convert a braille character represented by a hexadecimal number
    into the appropriate unicode character.
    Provided to students. Do not edit this function.

    >>> hex_to_unicode('2800')
    '⠀'
    >>> hex_to_unicode('2813')
    '⠓'
    >>> hex_to_unicode('2888')
    '⢈'
    '''
    # source: https://stackoverflow.com/questions/49958062/how-to-print-unicode-like-uvariable-in-python-2-7
    return chr(int(str(n),16))


def is_ostring(s):
    '''(str) -> bool
    Is s formatted like an o-string? It can be 6-dot or 8-dot.
    TODO: For students to complete.

    >>> is_ostring('o.\\noo\\n..')
    True
    >>> is_ostring('o.\\noo\\n..\\noo')
    True
    >>> is_ostring('o.\\n00\\n..\\noo')
    False
    >>> is_ostring('o.\\noo')
    False
    >>> is_ostring('o.o\\no\\n..')
    False
    >>> is_ostring('o.\\noo\\n..\\noo\\noo')
    False
    >>> is_ostring('\\n')
    False
    >>> is_ostring('A')
    False
    '''
    #create a constant 'symbol'
    #containg all the possible charater for an o-string
    symbol = 'o.\n'
    #count the num of character in line 1
    num_of_char_in_line_1 = s.find('\n')
    #count the num of character in line 2
    num_of_char_in_line_2 = s.find('\n',4)
    #count the num of character in line 2
    num_of_char_in_line_3 = s.find('\n',6)
    #count how num of lines 
    num_new_line = s.count('\n')
    #print(num_new_line)
    #creat constsnt check and assign it to True
    #so that it can firsly get into the loop which is used to
    #check the membership of char in s
    check = True
    #print(len(s))
    #checking for total length of str s
    if ((len(s)==8) or (len(s)==11)):
       
        #check num of new lines
        if (num_new_line == 2 or num_new_line == 3):
            
            #check num of char in line 1
            if (num_of_char_in_line_1 == 2):
                #check num of char in line 2
                if (num_of_char_in_line_2  == 5):
                    #check num of char in line 3
                    if (num_of_char_in_line_3 == s.find('\n',6)):
                        #the num of char in line 4 (if have) can be ensure
                        #by checking num in line 1,2,3 and total length
                        #if the test for total length passes
                        for i in range (len(s)):
                            if check:
                                
                            #creat a varibale 'check'
                            #to hold the updated boolean
                            #when cheking each charcter in s
                                check = s[i] in symbol
                                #to make sure the all characters in s
                                #have checked their membership in 'symbol'
                                if i == len(s)-1:
                                    return True
                                
    return False

def ostring_to_unicode(s):
    '''
    (str) -> str
    If s is a Braille cell in o-string format, convert it to unicode.
    Else return s.

    Remember from page 4 of the pdf:
    o-string -> raisedpos -> binary -> hex -> Unicode

    TODO: For students to complete.

    >>> ostring_to_unicode('o.\\noo\\n..')
    '⠓'
    >>> ostring_to_unicode('o.\\no.\\no.\\noo')
    '⣇'
    >>> ostring_to_unicode('oo\\noo\\noo\\noo')
    '⣿'
    >>> ostring_to_unicode('oo\\noo\\noo')
    '⠿'
    >>> ostring_to_unicode('..\\n..\\n..')
    '⠀'
    >>> ostring_to_unicode('a')
    'a'
    >>> ostring_to_unicode('\\n')
    '\\n'
    '''
    #check if s is in the form if o-string
    if is_ostring(s):
        #convert o-sting to raisedpos form
        raisedpos_form = ostring_to_raisedpos(s)
        #convert raisedpos form to binary
        bianry_form = raisedpos_to_binary(raisedpos_form)
        #convert binary form to hexademical number
        hex_form = binary_to_hex(bianry_form)
        #convert hexadeccimal form to unicode
        unicode = hex_to_unicode(hex_form)
        return unicode
    # if s is not in the form if o-string, return itself
  
    return s
#doctest.testmod()
a=ostring_to_unicode('o.\n..\n..')
b=ostring_to_unicode('o.\no.\n..')
c=ostring_to_unicode('oo\n..\n..')
d=ostring_to_unicode('oo\n.o\n..')
e=ostring_to_unicode('o.\n.o\n..')
f=ostring_to_unicode('oo\no.\n..')
g=ostring_to_unicode('oo\noo\n..')
h=ostring_to_unicode('o.\noo\n..')
i=ostring_to_unicode('.o\no.\n..')
j=ostring_to_unicode('.o\noo\n..')
k=ostring_to_unicode('o.\n..\no.')
l=ostring_to_unicode('o.\no.\no.')
m=ostring_to_unicode('oo\n..\no.')
n=ostring_to_unicode('oo\n.o\no.')
o=ostring_to_unicode('o.\n.o\no.')
p=ostring_to_unicode('oo\no.\no.')
q=ostring_to_unicode('oo\noo\no.')
r=ostring_to_unicode('o.\noo\no.')
s=ostring_to_unicode('.o\no.\no.')
t=ostring_to_unicode('.o\noo\no.')
u=ostring_to_unicode('o.\n..\noo')
v=ostring_to_unicode('o.\no.\noo')
w=ostring_to_unicode('.o\noo\n.o')
x_barille=ostring_to_unicode('oo\n..\noo')
y_barille=ostring_to_unicode('oo\n.o\noo')
z=ostring_to_unicode('o.\n.o\noo')
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
print(h)
print(i)
print(j)
print(k)
print(l)
print(m)
print(n)
print(o)
print(p)
print(q)
print(r)
print(s)
print(t)
print(u)
print(v)
print(w)
print(x_barille)
print(y_barille)
print(z)


# In[87]:


dataset = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')
dataset.rename(columns={'0':'label'}, inplace=True)


# In[88]:


print("Amount of each labels")

# Change label to letters
alphabets_mapper = {0.0:'A',1.0:'B',2.0:'C',3.0:'D',4.0:'E',5.0:'F',6.0:'G',7.0:'H',8.0:'I',9.0:'J',10.0:'K',11.0:'L',12.0:'M',13.0:'N',14.0:'O',15.0:'P',16.0:'Q',17.0:'R',18.0:'S',19.0:'T',20.0:'U',21.0:'V',22.0:'W',23.0:'X',24.0:'Y',25.0:'Z'}
print("before")
print(dataset.head())

# change to barille
barille_mapper = {'A':a,'B':b,'C':c,'D':d,'E':e,'F':f,'G':g,'H':h,'I':i,'J':j,'K':k,'L':l,'M':m,'N':n,'O':o,'P':p,'Q':q,'R':r,'S':s,'T':t,'U':u,'V':v,'W':w,'X':x,'Y':y,'Z':z} 
barille_mapper2 = {0.0:a,1.0:b,2.0:c,3.0:d,4.0:e,5.0:f,6.0:g,7.0:h,8.0:i,9.0:j,10.0:k,11.0:l,12.0:m,13.0:n,14.0:o,15.0:p,16.0:q,17.0:r,18.0:s,19.0:t,20.0:u,21.0:v,22.0:w,23.0:x_barille,24.0:y_barille,25.0:z} 
#print(type(dataset['label']))

#dataset_alphabets = dataset.copy()
dataset['label'] = dataset['label'].map(barille_mapper2)
print(dataset.head())
label_size = dataset.groupby('label').size()

#print(dataset.groupby('label').head())
label_size.plot.barh(figsize=(10,10))
plt.show()

#print("We have very low observations for I and F ")
#print("I count:", label_size['I'])
#print("F count:", label_size['F'])


# In[98]:


# splite the data


X_train, X_test, y_train, y_test = train_test_split(X,y_label)

# scale data
standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train)

X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)


# In[99]:


print("Data after scaler")
X_shuffle = shuffle(X_train)

plt.figure(figsize = (12,10))
row, colums = 4, 4
for i in range(16):  
    plt.subplot(colums, row, i+1)
    plt.imshow(X_shuffle[i].reshape(28,28),interpolation='nearest', cmap='Greys')
plt.show()


# In[100]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[102]:


# build the CNN
cls = Sequential()
cls.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
cls.add(MaxPooling2D(pool_size=(2, 2)))
cls.add(Dropout(0.3))
cls.add(Flatten())
cls.add(Dense(128, activation='relu'))
cls.add(Dense(len(y_label.unique()), activation='softmax'))

cls.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = cls.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=18, batch_size=200, verbose=2)

scores = cls.evaluate(X_test,y_test, verbose=0)
print("CNN Score:",scores[1])


# In[103]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[104]:


cm=confusion_matrix(y_test.argmax(axis=1),cls.predict(X_test).argmax(axis=1))
#print(cm)
df_cm = pd.DataFrame(cm, range(26),
                  range(26))
plt.figure(figsize = (20,15))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size


# In[105]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install -U coremltools')
import coremltools


# In[27]:


get_ipython().system('pip install tensorflowjs ')


# In[28]:


get_ipython().system('mkdir model')
get_ipython().system('tensorflowjs_converter --input_format keras my_model.h5 model/')


# In[106]:


cls.save('my_model.h5')


# In[107]:


get_ipython().system('zip -r model.zip model ')


# In[33]:



output_labels = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x_barill,y_barill,z]
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




