import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imsave, imshow
import utils

import model, os

# model = single_model(input_shape=(None, None) + (config.input_channels,), kernel_size=6)

model = model.deep_denoise_model(input_shape=(None, None, 3))
config = utils.MyConfig(type="deep_denoise", epochs=250, batch_size=16)

try:
    model.load_weights(config.model_path, by_name=True)
except Exception as inst:
    print inst
    os.remove(config.model_path)

model.summary()

# x_fns,y_fns=utils.gen_from_dir(config, mode=False)
# for x_fn in x_fns:
#     corr_img = imread(x_fn, mode='RGB')
#
#     plt.figure()
#     plt.imshow(corr_img)
#
#     x=utils.path2x(x_fn)
#     x=x[np.newaxis,...]
#
#     y=model.predict(x)
#
#     restore_img = utils.y2img(y[0])
#
#     plt.figure()
#     plt.imshow(restore_img)
#
#     plt.show()

x_fns, y_fns = utils.common_paths(config.test_X_path, config.test_y_path, config)
for x_fn, y_fn in zip(x_fns, y_fns):
    corr_img = imread(x_fn, mode='RGB')
    ori_img = imread(y_fn, mode='RGB')

    plt.figure()
    plt.imshow(corr_img)
    plt.figure()
    plt.imshow(ori_img)

    x = utils.path2x(x_fn)
    assert x.shape[-1] == 3

    x = utils.make_patches(x, patch_size=16)
    y = model.predict(x, batch_size=1024, verbose=0)
    y = utils.combine_patches(y, out_shape=ori_img.shape)
    y = utils.y2img(y)

    restore_img = np.clip(y, 0, 255).astype('uint8')
    # restore_img = utils.conserve_img(y, corr_img)
    restore_img=utils.post_process(ori_img, restore_img)
    import os

    img_restore_name = config.test_yo_path + os.path.basename(x_fn)
    plt.imsave(img_restore_name, restore_img)
    plt.figure()
    plt.imshow(restore_img,cmap='gray' if len(restore_img.shape)==2 else None)

    plt.show()
