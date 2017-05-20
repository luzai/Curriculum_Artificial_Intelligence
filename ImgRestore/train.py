import glob
import keras
import matplotlib.pyplot as plt
import numpy as np
import os

from model import single_model
from utils import MyConfig, img2x, norm_img

config = MyConfig()


def gen_x_y(ind, img_l):
    img_name = img_l[ind]
    img = plt.imread(img_name)
    x = img2x(img)
    x = norm_img(x)
    ori_img_name = os.path.join(
        config.ori_prefix,
        os.path.basename(img_name)
    )

    y = plt.imread(ori_img_name)
    y = norm_img(y)
    return x, y


def gen_from_dir(config):
    while 1:
        config.update()
        img_l = glob.glob(
            os.path.join(config.corr_prefix, '*.' + config.suffix)
        )
        img_l = np.random.permutation(img_l)
        X = np.ones((config.batch_size,) + config.train_img_shape + (config.input_channels,), dtype=np.float)
        Y = np.ones((config.batch_size,) + config.train_img_shape + (config.output_channels,), dtype=np.float)
        for ind, img_name in enumerate(img_l):

            img = plt.imread(img_name)
            x = img2x(img)
            x = norm_img(x)
            X[ind % config.batch_size] = x

            ori_img_name = os.path.join(
                config.ori_prefix,
                os.path.basename(img_name)
            )

            y = plt.imread(ori_img_name)
            y = norm_img(y)
            Y[ind % config.batch_size] = y

            assert not np.array_equal(x, y), "not same"

            if (ind + 1) % config.batch_size == 0:
                yield (X, Y)


from model import deep_model
# model = single_model(input_shape=(None, None) + (config.input_channels,), kernel_size=20)
model = deep_model(input_shape=(None, None) + (config.input_channels,))

model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mse'])
model_name='deep.h5'
try:
    model.load_weights(config.sav_path+model_name, by_name=True)
except Exception as inst:
    print inst
    # import h5py
    #
    # with h5py.File(config.sav_path, 'r') as f:
    #     # X_data = f['X_data']
    #     pass

model.summary()

model.fit_generator(gen_from_dir(config),
                    steps_per_epoch=config.imgs // config.batch_size,
                    epochs=2
                    ,callbacks=[keras.callbacks.ModelCheckpoint(config.sav_path, monitor='train_loss')]
                    )
model.save(config.sav_path+model_name)
