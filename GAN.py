import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist.input_data import input_data
from keras.models import Dense,Activation


def build_generator(latent_size):
    cnn = Sequential()
    cnn.add(Dense(1024,input_dim=latent_size,activation='relu'))
    cnn.add(Dense(128*7*7,activation ='relu'))
    cnn.add(Reshape((128,7,7)))
    cnn.add(Upsampling2D(size=(2,2)))
    cnn.add(Convolution2D(256,5,5,border_model ='same',activation ='relu',init ='glorot_normal'))

    cnn.add(Upsampling2D(size=(2,2)))
    cnn.add(Convolution2D(128,5,5,border_model='same',activation='relu',init = 'glorot_normal'))

    cnn.add(Convolution2D(1,2,2,border_model='same',activation='relu',init ='glorot_normal'))

    latent = Input(shape=(latent_size, ))

    image_class = Input(shape = (1, ),dtype = 'int32')
    cls = Flatten(Embedding(10,latent_size,init = 'glorot_normal',image_class))
    h = merge([latent,cls],mode ='mul')
    fake_image = cnn(h)
    
    return Model(input=[laten,image_class],output=fake_image)
