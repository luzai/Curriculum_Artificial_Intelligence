import sys
import sys,os
std_out=sys.stdout
std_err= open('stderr', 'w')
sys.stdout=std_err
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import model as MyModels
import utils, os, multiprocessing
import time,cv2
import logging,sys

name='deep_denoise'
config = utils.MyConfig(type=name,epochs=3,batch_size=1024,verbose=2)
model=MyModels.__dict__[name+'_model'](input_shape=(None,None)+(config.input_channels,))
# consider todo x shufle

try:
    # os.remove(config.model_path)
    model.load_weights(config.model_path, by_name=True)
except Exception as inst:
    print inst
    exit(-2)
    # os.remove(config.model_path)
sys.stdout=std_out
if config.verbose: model.summary()

callback_list = [
    keras.callbacks.EarlyStopping(monitor='val_loss2acc', min_delta=0.1, patience=2)
]

x_fns, y_fns = utils.common_paths(config.test_X_path, config.test_y_path, config)
import cv2
for x_fn ,y_fn in zip(x_fns,y_fns):
    print x_fn,y_fn

    x=imread(x_fn,mode='RGB')
    y=imread(y_fn,mode='RGB')
    print x.shape
    print utils.my_mse(x,y)

    # print x
    # assert np.max(x)>2.
    # y=imread(y_fn,mode='RGB')
    # y_pred=cv2.medianBlur(x,ksize=3)
    # my_imshow = lambda y_pred,name:  utils.my_imshow(y_pred,block=False,name=name)
    # my_imshow(y_pred,'y_pred')
    # my_imshow(y,'y')
    # my_imshow(x,'x')
x_fns, y_fns = utils.common_paths(config.val_X_path, config.val_y_path, config)
print '-'*10
for x_fn, y_fn in zip(x_fns, y_fns):
    print x_fn, y_fn

    x = imread(x_fn, mode='RGB')
    y = imread(y_fn, mode='RGB')
    print x.shape
    print utils.my_mse(x, y)
print '-'*10
# x_fns, y_fns = utils.common_paths(config.train_X_path, config.train_y_path, config)
# for x_fn, y_fn in zip(x_fns, y_fns):
#     print x_fn, y_fn
#
#     x = imread(x_fn, mode='RGB')
#     y = imread(y_fn, mode='RGB')
#     print x.shape
#     print utils.my_mse(x, y)