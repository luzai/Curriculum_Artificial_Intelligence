
import os,glob,sys,subprocess,pprint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.backend as K

def mse(x,y):
    return ((x-y)**2).mean(axis=None)

def conserve_img(x,y):
    mask_dd=(x!=0).astype('uint8')
    assert mask_dd.shape==y.shape,'shape same'
    y[mask_dd]=x[mask_dd]
    return  y

def denorm_img(img):
    img*=255
    img=img.astype('uint8')
    return img

def img2x(img):
    mask = (img == 0).astype('uint8')
    assert mask.shape[-1] == 3, "mask is 3 channel"
    x = np.concatenate((img, mask), axis=2)
    assert x.shape[-1] == 6, 'x is 6 channels'
    # x = x[:config.train_img_shape[0], :config.train_img_shape[1], :]
    return x

def gray2rgb(img):
    res= np.stack((img,img.copy(),img.copy()),axis=2)
    if np.max(res)<255:
        res*=255
    return res.astype('uint8')
def norm_img(img):
    img=img.astype(float)
    img/=255.
    return img

class MyConfig(object):
    ori_prefix='data/voc2012_ori/'
    corr_prefix='data/voc2012_corr/'
    suffix='jpg'
    train_img_shape=(512, 512)
    batch_size=1
    input_channels=6
    output_channels=3
    sav_path= 'output/'
    tf_graph=tf.get_default_graph()
    _sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        # log_device_placement = True,
        # inter_op_parallelism_threads = 8,
        # intra_op_parallelism_threads = 8
    )
    _sess_config.gpu_options.allow_growth = True
    # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=_sess_config, graph=tf_graph)
    K.set_session(sess)
    K.set_image_data_format("channels_last")

    def __init__(self):
        self.update()

    def update(self):
        img_l = glob.glob(
            os.path.join(self.ori_prefix, '*.' + self.suffix)
        )

        self.imgs = len(img_l)

