import os, glob, sys, subprocess, pprint
import tensorflow as tf
import matplotlib

matplotlib.use('TKAgg')
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


def my_mse(x, y):
    if len(x.shape) == 2 and len(y.shape)==3:
        y=y.mean(axis=-1)
    elif len(y.shape) == 2 and len(x.shape)==3:
        x=x.mean(axis=-1)

    assert len(x.shape) == len(y.shape), 'dims should same'
    x, y = x.astype('float'), y.astype('float')
    if x.max() > 2.:
        x /= 255.
    if y.max() > 2.:
        y /= 255.
    from numpy import linalg as LA
    res = LA.norm(x.ravel() - y.ravel(), 2)
    return res


# def path2x(path):
#     img = imread(path, mode="RGB")
#     return img2x(img)


def get_mask(x, bool=False):
    if bool:
        return (x != 0).astype('bool')
    else:
        return (x != 0).astype('uint8')  # 0 means missing


def img2x(img, config,patch_size=8):
    assert np.max(img) > 2.
    img_01 = img.astype('float32') / 255.
    res = []
    if config.rgb_in:
        res += [img_01]
    if config.pos_in:
        #  todo
        pass
    if config.mask_in:
        mask = get_mask(img)
        res += [mask]
    res = np.concatenate(res, axis=2)
    if not config.train:
        res = make_patches(res, patch_size=patch_size)

    else:
        res = res[np.newaxis, ...]
        assert len(res.shape) == 4

    return res


import time


def y2img(restore_img, corr_img, config=None):
    assert np.max(restore_img) < 2., 'assert fail {}'.format(np.max(restore_img))
    assert np.max(corr_img) > 2.
    # print  time.time()
    if len(restore_img.shape) == 4:
        restore_img = combine_patches(restore_img, corr_img.shape)
    # print  time.time()
    restore_img = (restore_img * 255.).astype('uint8')
    restore_img = np.clip(restore_img, 0, 255).astype('uint8')

    # print  time.time()
    restore_img = post_process(x_from=corr_img, y_to=restore_img)
    # print  time.time()
    return restore_img


def post_process(x_from, y_to):
    assert x_from.shape == y_to.shape, 'shape same'
    mask_dd = (x_from != 0).astype('bool')
    assert mask_dd.shape == y_to.shape, 'shape same'
    y_to[mask_dd] = x_from[mask_dd]

    # if np.array_equal(x_from[..., 0], x_from[..., 1]):
    #     y_to = y_to.mean(axis=2)

    return y_to


def make_patches(x, patch_size):
    # height, width = x.shape[:2]
    patches = extract_patches_2d(x, (patch_size, patch_size))
    return patches


def combine_patches(y, out_shape):
    recon = reconstruct_from_patches_2d(y, out_shape)
    return recon


import threading


class ReadData(threading.Thread):
    def __init__(self, X_name, ind, X, lock, config):
        self.X_name = X_name
        self.ind = ind
        self.X = X
        self.config = config
        self.lock = lock
        super(ReadData, self).__init__()

    def run(self):
        with self.lock:
            if self.X.shape[-1] > 3:
                x = imread(self.X_name, mode='RGB')
                x = img2x(x, self.config)
                self.X[self.ind] = x
            else:
                y = imread(self.X_name, mode='RGB')
                y = y.astype('float32') / 255.
                self.X[self.ind] = y


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
    else:
        X_filenames, y_filenames = val_paths(config)
    assert len(X_filenames) == len(y_filenames)
    nb_images = len(X_filenames)
    index_gen = _index_generator(nb_images, config.train_batch_size)

    lock = threading.Lock()

    index_array, current_index, current_batch_size = next(index_gen)

    X = np.ones((config.train_batch_size,) + config.train_img_shape + (config.input_channels,), dtype=np.float)
    Y = np.ones((config.train_batch_size,) + config.train_img_shape + (config.output_channels,), dtype=np.float)
    threads = []

    for i, j in enumerate(index_array):
        x_fn = X_filenames[j]
        tx = ReadData(x_fn, i, X, lock, config)
        tx.start()
        threads.append(tx)
        y_fn = y_filenames[j]
        ty = ReadData(y_fn, i, Y, lock, config)
        ty.start()
        threads.append(ty)

    for t in threads:
        t.join()

    cache = (X, Y)
    assert X.max() < 2. and Y.max() < 2.
    while 1:

        index_array, current_index, current_batch_size = next(index_gen)

        X = np.ones((config.train_batch_size,) + config.train_img_shape + (config.input_channels,), dtype=np.float)
        Y = np.ones((config.train_batch_size,) + config.train_img_shape + (config.output_channels,), dtype=np.float)
        threads = []

        for i, j in enumerate(index_array):
            x_fn = X_filenames[j]
            tx = ReadData(x_fn, i, X, lock, config)
            tx.start()
            threads.append(tx)
            y_fn = y_filenames[j]
            ty = ReadData(y_fn, i, Y, lock, config)
            ty.start()
            threads.append(ty)

        yield cache

        for t in threads:
            t.join()
        cache = (X, Y)
        assert X.min() < 2. and Y.min() < 2.


def get_steps(config, train=True):
    if train:
        x_fn, _ = train_paths(config)
    else:
        x_fn, _ = val_paths(config)
    steps = len(x_fn) // config.train_batch_size
    return int(steps)


def train_paths(config):
    return common_paths(config.train_X_path, config.train_y_path, config)


def val_paths(config):
    return common_paths(config.val_X_path, config.val_y_path, config)


def common_paths(x_path, y_path, config):
    file_names = [f for f in sorted(os.listdir(x_path), reverse=True)
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

    suffixs = ['png']  # , 'jpg']
    train_img_shape = (256, 256)

    output_channels = 3

    tf_graph = tf.get_default_graph()
    _sess_config = tf.ConfigProto(
        allow_soft_placement=True,
    )
    _sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=_sess_config, graph=tf_graph)
    K.set_session(sess)
    K.set_image_data_format("channels_last")

    def __init__(self, type="deep_denoise", rgb_in=True, pos_in=False, train_epochs=None, train_batch_size=None,
                 epochs=3, batch_size=1024, verbose=2):
        self.verbose = verbose
        self.epochs = epochs
        if train_batch_size is not None:
            self.train = True
            self.train_epochs = train_epochs
            self.train_batch_size = train_batch_size
        else:
            self.train = False
            self.epochs = epochs
            self.batch_size = batch_size
        self.input_channels = 3 * (int(rgb_in) + int(pos_in) + 1)

        self.rgb_in = rgb_in
        if rgb_in:
            type += '_rgb'
        self.pos_in = pos_in
        if pos_in:
            type += '_pos'
        self.mask_in = True

        self.model_name = type + ".h5"
        self.model_path = "output/" + self.model_name


def my_imshow(img, cmap=None, block=False, name='default'):
    if block:
        fig, ax = plt.subplots()
        if len(img.shape) == 3 and img.shape[2] == 3 and img.max() > 2.:
            img = img.astype('uint8')
        ax.imshow(img, cmap)
        ax.set_title(name)
        fig.canvas.set_window_title(name)
        plt.show()
    else:
        import multiprocessing
        multiprocessing.Process(target=my_imshow, args=(img, cmap, True, name)).start()


def my_dbg():
    from IPython import embed;
    embed()




if __name__ == "__main__":
    import time


    config = MyConfig(type="deep_denoise", train_epochs=2, train_batch_size=16)
    print len(train_paths(config)[1])
    print len(val_paths(config)[1])
    for x, y in gen_from_dir(config, mode=True):
        print x.shape, y.shape, time.time()
        xt = x[0][..., :3]
        xt = (xt * 255).astype('uint8')
        yt = y[0][..., :3]
        yt = (yt * 255).astype('uint8')
        my_imshow(xt)
        my_imshow(yt, name='yt')
        break

