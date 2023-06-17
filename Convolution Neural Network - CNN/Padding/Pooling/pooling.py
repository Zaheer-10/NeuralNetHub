import keras 
from keras.models import Sequential 
from keras.layers import Activation 
from keras.layers.core import Dense , Flatten
from keras.layers.convolutional import *
from keras.layers.pooling import * 

model = Sequential ([
    Dense (16 , activation = 'relu' , input_shape = (20,20,3)),
    Conv2D (32 , kernel_size = (3,3) , activation = 'relu' , padding = 'same'),

    MaxPooling2D(pool_size= (2,2) , strides = 2 , padding = 'valid'),

    Conv2D (64 , kernel_size = (5,5) , activation = 'relu' , padding = 'same'),
    Flatten(),
    Dense(2 , activation = 'softmax')


])

model.summary()

'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_2 (Dense)             (None, 20, 20, 16)        64        
                                                                 
 conv2d (Conv2D)             (None, 20, 20, 32)        4640      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 10, 10, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 10, 10, 64)        51264     
                                                                 
 flatten (Flatten)           (None, 6400)              0         
                                                                 
 dense_3 (Dense)             (None, 2)                 12802     
                                                                 
=================================================================
Total params: 68,770
Trainable params: 68,770
Non-trainable params: 0
_________________________________________________________________
'''