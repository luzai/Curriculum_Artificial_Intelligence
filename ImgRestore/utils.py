import os, glob, sys, subprocess, pprint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.backend as K
from scipy.misc import imread, imsave, imshow
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d

def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)
    return model


def mse(x, y):
    if len(x.shape) == 2 and len(y.shape) == 3:
        y = y.mean(axis=2)
    elif len(y.shape) == 2 and len(x.shape) == 3:
        x = x.mean(axis=2)
    return ((x - y) ** 2).mean(axis=None)


def path2x(path):
    img = imread(path, mode="RGB")
    return img2x(img)


def img2x(img):
    assert np.max(img) > 2.
    return img.astype('float32') / 255.


def y2img(y):
    assert np.max(y) < 2.
    return (y * 255.).astype('uint8')


def get_mask(x):
    return (x != 0).astype('uint8')  # 0 means missing


def post_process(x, y):
    assert x.shape == y.shape, 'shape same'
    mask_dd = (x != 0).astype('uint8')
    assert mask_dd.shape == y.shape, 'shape same'
    y[mask_dd] = x[mask_dd]

    if np.array_equal(x[..., 0], x[..., 1]):
        y = y.mean(axis=2)

    return y


def make_patches(x, patch_size):
    # height, width = x.shape[:2]
    patches = extract_patches_2d(x, (patch_size, patch_size))
    return patches


def combine_patches(y, out_shape):
    recon = reconstruct_from_patches_2d(y, out_shape)
    return recon


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


cache = None


def gen_from_dir(config, mode=True):
    global cache
    if mode == True:
        X_filenames, y_filenames = train_paths(config)
    elif mode == False:
        X_filenames, y_filenames = val_paths(config)
    # elif mode== 'test':
    #     X_filenames, y_filenames = val_paths(config)
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

    cache = (X, Y)

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
        cache = (X, Y)


def get_steps(config, train=True):
    if train:
        x_fn, _ = train_paths(config)
    else:
        x_fn, _ = val_paths(config)
    steps = len(x_fn) // config.batch_size
    return int(steps)


def train_paths(config):
    return common_paths(config.train_X_path, config.train_y_path, config)


def val_paths(config):
    return common_paths(config.val_X_path, config.val_y_path, config)


def common_paths(x_path, y_path, config):
    file_names = [f for f in sorted(os.listdir(x_path))
                  if np.array([f.endswith(suffix) for suffix in config.suffixs]).any()]
    X_filenames = [os.path.join(x_path, f) for f in file_names]
    y_filenames = [os.path.join(y_path, f) for f in file_names]

    return X_filenames, y_filenames


class MyConfig(object):
    train_y_path = 'data/voc2012_ori/'
    train_X_path = 'data/voc2012_corr/'
    val_X_path = 'data/val_corr/'
    val_y_path = 'data/val_ori/'
    test_X_path = 'data/test_corr/'
    test_y_path = 'data/test_ori/'
    test_yo_path = 'data/test_restore/'

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

    def __init__(self, type="deep_denoise",rgb_in=True,pos_in=False,epochs=20, batch_size=6, input_channels=3,train=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_channels = 3 * (int(rgb_in)+int(pos_in)+1)
        if rgb_in :
            type+='_rgb'
        elif pos_in:
            type+='_pos'
        self.model_name = type  + ".h5"
        self.model_path = "output/" + self.model_name

if __name__ == "__main__":
    import time

    config = MyConfig(type="deep_denoise", epochs=2, batch_size=16)
    print len(train_paths(config)[1])
    print len(val_paths(config)[1])
    for x, y in gen_from_dir(config, mode=True):
        print x.shape, y.shape, time.time()
        imshow(y2img(x[0]))
        imshow(y2img(y[0]))
        break
