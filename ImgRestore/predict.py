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
for iter_ind, (x_fn, y_fn) in enumerate(zip(x_fns, y_fns)):
    model=utils.shuffle_weights(model)
    corr_img = imread(x_fn, mode='RGB')
    ori_img = imread(y_fn, mode='RGB')
    if np.array_equal(corr_img[..., 0], corr_img[..., 1]):
        continue # todo

    x=utils.img2x(corr_img,config)

    y_true = x.copy()

    model.fit(x, y_true, batch_size=config.batch_size,
              epochs=config.epochs,#*2 if iter_ind==0 else config.epochs,
              verbose=config.verbose, callbacks=callback_list,
              validation_split=0.1)
    y=model.predict(x)
    restore_img = utils.y2img(y,config,corr_img)
    cmap = 'gray' if len(restore_img.shape) == 2 else None

    print 'plt start'
    utils.my_imshow(restore_img, cmap,block=False,name='restore')
    utils.my_imshow(corr_img, cmap,block=False,name='corr')

    print(utils.my_mse(ori_img, restore_img))
    # from IPython import  embed;embed()
    # model.save(config.model_path)
