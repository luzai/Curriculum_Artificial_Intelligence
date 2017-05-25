import glob

import matplotlib.pyplot as plt
import numpy as np

import utils
from model import single_model
import model

# model = single_model(input_shape=(None, None) + (config.input_channels,), kernel_size=6)
# model.load_weights(config.sav_path+'model.h5')
model = model.deep_denoise_model(input_shape=(None,None,3))
config = utils.MyConfig(type="deep_denoise", epochs=250,batch_size=16)

model.summary()

x_fns,y_fns=utils.common_paths('data/test_corr/','data/test_ori')

for img_name in img_l:

    img = plt.imread(img_name)

    x = img.copy()
    if not x.shape[-1] == 3:
        x = gray2rgb(x)
    x = utils.denorm_img(x)
    assert x.dtype == np.uint8
    plt.figure()
    plt.imshow(x)

    assert x.shape[-1] == 3
    x = img2x(x)
    if img.shape[0] % 4 != 0:
        x = np.concatenate((x[:(4 - img.shape[0] % 4), :, :].copy(), x), axis=0)
    if img.shape[1] % 4 != 0:
        x = np.concatenate((x[:, :(4 - img.shape[1] % 4), :].copy(), x), axis=1)
    x = utils.norm_img(x)
    print x.shape

    y = model.predict(x[np.newaxis, ...])[0]
    if img.shape[0] % 4 != 0:
        y = y[(4 - img.shape[0] % 4):, :, :]
    if img.shape[1] % 4 != 0:
        y = y[:, (4 - img.shape[1] % 4):, :]
    y = utils.denorm_img(y)
    assert y.dtype == np.uint8

    if not img.shape[-1] == 3:
        y = utils.rgb2gray(y)
    y = utils.conserve_img(img, y)

    import os

    img_ori_name = 'data/test_ori/' + os.path.basename(img_name)
    img_ori = plt.imread(img_ori_name)
    img_ori = utils.denorm_img(img_ori)
    print utils.mse(img_ori, y)

    if img.shape[-1] == 3:
        plt.figure()
        plt.imshow(y)
        plt.figure()
        plt.imshow(img_ori)
    else:
        plt.figure()
        plt.imshow(y, cmap='gray')
        plt.figure()
        plt.imshow(img_ori, cmap='gray')
    img_restore_name = 'data/test_restore/' + os.path.basename(img_name)
    plt.imsave(img_restore_name, y)
plt.show()
