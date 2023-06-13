from keras.layers.merging.average import Average
import keras 
from keras.models import Sequential 
from keras.layers import Activation 
from keras.layers.core import Dense , Flatten
from keras.layers.convolutional import *
from keras.layers.pooling import * 


model = Sequential ()

model.add(Conv2D(6 , kernel_size = (5,5) , padding = 'valid' , activation= 'tanh' , input_shape=(32,32,1)))
model.add(AveragePooling2D(pool_size=(2,2) , strides=2 , padding = 'valid'))

model.add(Conv2D(16 , kernel_size = (5,5) , padding = 'valid' , activation= 'tanh' ))
model.add(AveragePooling2D(pool_size=(2,2) , strides=2 , padding = 'valid'))

model.add(Flatten())

model.add(Dense(120 , activation = 'tanh'))
model.add(Dense(84 , activation = 'tanh'))
model.add(Dense(10 , activation = 'softmax'))

model.summary()

'''
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_2 (Conv2D)           (None, 28, 28, 6)         156       
                                                                 
 average_pooling2d_2 (Average  (None, 14, 14, 6)        0         
 ePooling2D)                                                     
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 16)        2416      
                                                                 
 average_pooling2d_3 (Average  (None, 5, 5, 16)         0         
 ePooling2D)                                                     
                                                                 
 flatten (Flatten)           (None, 400)               0         
                                                                 
 dense (Dense)               (None, 120)               48120     
                                                                 
 dense_1 (Dense)             (None, 84)                10164     
                                                                 
 dense_2 (Dense)             (None, 10)                850       
                                                                 
=================================================================
Total params: 61,706
Trainable params: 61,706
Non-trainable params: 0
_________________________________________________________________
'''