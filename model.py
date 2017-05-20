from utils import MyConfig
import os,glob,sys,subprocess,pprint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.backend as K


def single_model(input_shape=(None, None, 6), kernel_size=5):
    input = keras.layers.Input(shape=input_shape)
    output = keras.layers.Conv2D(filters=3, kernel_size=kernel_size, padding='same', activation='relu',name='conv1')(input)
    model = keras.models.Model(inputs=input, outputs=output)
    return model

def deep_model(input_shape=(None, None, 6)):
    input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',name='conv1')(input)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',name='conv2')(input)
    output=x
    model = keras.models.Model(inputs=input, outputs=output)

if __name__ == '__main__':
    model = single_model()
