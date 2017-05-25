import os, glob, sys, subprocess, pprint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.backend as K
from scipy.misc import imread, imsave, imshow


def mse(x, y):
    return ((x - y) ** 2).mean(axis=None)


def path2x(path):
    img = imread(path, mode="RGB")
    img = img.astype('float32') / 255.
    return img


def y2img(y):
    y *= 255.
    y.astype('uint8')
    return y


# def conserve_img(x, y):
#     assert x.shape == y.shape, 'shape same'
#     mask_dd = (x != 0).astype('uint8')
#     assert mask_dd.shape == y.shape, 'shape same'
#     y[mask_dd] = x[mask_dd]
#     return y
#
#
# def img2x(img):
#     mask = (img == 0).astype('uint8')
#     assert mask.shape[-1] == 3, "mask is 3 channel"
#     x = np.concatenate((img, mask), axis=2)
#     assert x.shape[-1] == 6, 'x is 6 channels'
#     # x = x[:config.train_img_shape[0], :config.train_img_shape[1], :]
#     return x
import threading


class ReadData(threading.Thread):
    def __init__(self, X_name, ind, X, lock):
        self.X_name = X_name
        self.ind = ind
        self.X = X
        self.lock = lock
        super(ReadData, self).__init__()

    def run(self):
        with self.lock:
            self.X[self.ind] = path2x(self.X_name)


def _index_generator(N, batch_size=32, shuffle=True, seed=None):
    batch_index = 0
    total_batches_seen = 0

    while 1:
        if seed is not None:
            np.random.seed(seed + total_batches_seen)

        if batch_index == 0:
            index_array = np.arange(N)
            if shuffle:
                index_array = np.random.permutation(N)

        current_index = (batch_index * batch_size) % N

        if N >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
        total_batches_seen += 1

        yield (index_array[current_index: current_index + current_batch_size],
               current_index, current_batch_size)

cache=None
def gen_from_dir(config, train=True):
    global cache
    if train:
        X_filenames, y_filenames = train_paths(config)
    else:
        X_filenames, y_filenames = val_paths(config)
    assert len(X_filenames) == len(y_filenames)
    nb_images = len(X_filenames)
    index_gen = _index_generator(nb_images, config.batch_size)

    lock = threading.Lock()

    index_array, current_index, current_batch_size = next(index_gen)

    X = np.ones((config.batch_size,) + config.train_img_shape + (config.input_channels,), dtype=np.float)
    Y = np.ones((config.batch_size,) + config.train_img_shape + (config.output_channels,), dtype=np.float)
    threads = []

    for i, j in enumerate(index_array):
        x_fn = X_filenames[j]
        tx = ReadData(x_fn, i, X, lock)
        tx.start()
        threads.append(tx)
        y_fn = y_filenames[j]
        ty = ReadData(y_fn, i, Y, lock)
        ty.start()
        threads.append(ty)

    for t in threads:
        t.join()

    cache=(X,Y)

    while 1:

        index_array, current_index, current_batch_size = next(index_gen)

        X = np.ones((config.batch_size,) + config.train_img_shape + (config.input_channels,), dtype=np.float)
        Y = np.ones((config.batch_size,) + config.train_img_shape + (config.output_channels,), dtype=np.float)
        threads = []

        for i, j in enumerate(index_array):
            x_fn = X_filenames[j]
            tx = ReadData(x_fn, i, X, lock)
            tx.start()
            threads.append(tx)
            y_fn = y_filenames[j]
            ty = ReadData(y_fn, i, Y, lock)
            ty.start()
            threads.append(ty)

        yield cache

        for t in threads:
            t.join()
        cache=(X,Y)


def get_steps(config,train=True):
    if train:
        x_fn, _ = train_paths(config)
    else:
        x_fn, _ = val_paths(config)
    steps = len(x_fn) // config.batch_size
    return int(steps)


def train_paths(config):
    file_names = [f for f in sorted(os.listdir(config.train_X_path))
                  if np.array([f.endswith(suffix) for suffix in config.suffixs]).any()]
    X_filenames = [os.path.join(config.train_X_path, f) for f in file_names]
    y_filenames = [os.path.join(config.train_y_path, f) for f in file_names]
    # X_filenames, y_filenames=val_paths(config)
    return X_filenames, y_filenames


def val_paths(config):
    # file_names = [f for f in sorted(os.listdir(config.val_X_path))
    #               if np.array([f.endswith(suffix) for suffix in config.suffixs]).any()
    #               ]
    # X_filenames = [os.path.join(config.val_X_path, f) for f in file_names]
    # y_filenames = [os.path.join(config.val_X_path, f) for f in file_names]
    X_filenames,y_filenames=train_paths(config)
    return X_filenames, y_filenames

def common_paths(x_path,y_path):
    file_names = [f for f in sorted(os.listdir(x_path))
                  if np.array([f.endswith(suffix) for suffix in config.suffixs]).any()]
    X_filenames = [os.path.join(x_path, f) for f in file_names]
    y_filenames = [os.path.join(y_path, f) for f in file_names]
    # X_filenames, y_filenames=val_paths(config)
    return X_filenames, y_filenames

class MyConfig(object):
    train_y_path = 'data/voc2012_ori/'
    train_X_path = 'data/voc2012_corr/'
    val_X_path = 'data/test_corr/'
    val_y_path = 'data/test_ori/'
    suffixs = ['jpg', 'png']
    train_img_shape = (512, 512)

    output_channels = 3

    tf_graph = tf.get_default_graph()
    _sess_config = tf.ConfigProto(
        allow_soft_placement=True,
    )
    _sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=_sess_config, graph=tf_graph)
    K.set_session(sess)
    K.set_image_data_format("channels_last")

    def __init__(self, type="deep_denoise", epochs=20, batch_size=6, input_channels=3):
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_channels = input_channels
        if type == "deep_denoise":
            self.model_name = type + ".h5"
            self.model_path = "output/" + self.model_name


if __name__ == "__main__":
    config = MyConfig(type="deep_denoise", epochs=2, batch_size=128)
    print len(train_paths(config)[1])
    print len(val_paths(config)[1])
    for x, y in gen_from_dir(config, train=True):
        print x.shape, y.shape
        imshow(x[0])
        imshow(y[0])
        break
