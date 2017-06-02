import matplotlib

matplotlib.use('TKAgg')
from scipy.misc import imread
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
import os
from keras.utils import vis_utils
from IPython.display import display, HTML, SVG

import keras.backend as K
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>" % size)
    return strip_def


def i_vis_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


def i_vis_model(model):
    SVG(vis_utils.model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
import os.path as osp
def vis_model(model, name='net2net', show_shapes=True):
    try:
        # vis_utils.plot_model(model, to_file=name + '.pdf', show_shapes=show_shapes)
        vis_utils.plot_model(model, to_file=name + '.png', show_shapes=show_shapes)
    except Exception as inst:
        print("cannot keras.plot_model {}".format(inst))



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
    if x.shape[-1] == 1:
        x = x.mean(axis=-1)
    if y.shape[-1] == 1:
        y = y.mean(axis=-1)
    if len(x.shape) == 2 and len(y.shape) == 3:
        y = y.mean(axis=-1)
    elif len(y.shape) == 2 and len(x.shape) == 3:
        x = x.mean(axis=-1)

    assert len(x.shape) == len(y.shape), 'dims should same'
    x, y = x.astype('float'), y.astype('float')
    if x.max() > 2.:
        x /= 255.
    if y.max() > 2.:
        y /= 255.
    from numpy import linalg as LA
    res = LA.norm(x.ravel() - y.ravel(), 2)
    return res


def get_mask(x, bool=False):
    if bool:
        return (x != 0).astype('bool')
    else:
        return (x != 0).astype('uint8')  # 0 means missing


def img2x(img, config, patch_size=8):
    assert np.max(img) > 2.
    img_01 = img.astype('float32') / 255.
    if 'gray' in config.type:
        img_01 = rgb2gray(img_01)
    res = []
    if config.rgb_in:
        res += [img_01]
    if config.pos_in:
        #  todo
        pass
    if config.mask_in:
        mask = get_mask(img)
        if 'gray' in config.type:
            mask = rgb2gray(mask)
        res += [mask]
    assert not np.array_equal(res[0], res[1])
    res = np.concatenate(res, axis=2)
    if not config.train:
        res = make_patches(res, patch_size=patch_size)
    else:
        res = res[np.newaxis, ...]
    assert len(res.shape) == 4

    return res


def y2img(restore_img, corr_img, config=None):
    assert np.max(restore_img) < 2., 'assert fail {}'.format(np.max(restore_img))
    assert np.max(corr_img) > 2.

    if len(restore_img.shape) == 4:
        if 'gray' not in config.type:
            shape = corr_img.shape
        else:
            shape = corr_img.shape[:-1] + (1,)
        # restore_img=restore_img[0]
        restore_img = combine_patches(restore_img, corr_img.shape)

    restore_img = (restore_img * 255.).astype('uint8')
    restore_img = np.clip(restore_img, 0, 255).astype('uint8')

    restore_img = post_process(x_from_in=corr_img, y_to=restore_img, config=config)

    return restore_img


def post_process(x_from_in, y_to, config=None):
    if 'gray' in config.type and x_from_in.shape[-1] == 3:
        x_from = rgb2gray(x_from_in)
    else:
        x_from = x_from_in
    # x_from = x_from.mean(axis=-1)
    # y_to = y_to.mean(axis=-1)
    assert x_from.shape == y_to.shape, 'shape same'
    mask_dd = (x_from != 0).astype('bool')
    assert mask_dd.shape == y_to.shape, 'shape same'
    y_to[mask_dd] = x_from[mask_dd]

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
                assert not np.array_equal(x[0], x[1])
                self.X[self.ind] = x
            else:
                y = imread(self.X_name, mode='RGB')
                if 'gray' in self.config.type:
                    y = rgb2gray(y)
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

        yield (X, Y)
        assert X.min() < 2. and Y.min() < 2.


def get_steps(config, train=True):
    if train:
        x_fn, _ = train_paths(config)
    else:
        x_fn, _ = val_paths(config)
    steps = len(x_fn) // config.train_batch_size
    return int(steps)


def train_paths(config):
    X_filenames, y_filenames = common_paths(config.train_X_path, config.train_y_path, config)
    ind = np.random.permutation(len(X_filenames)).tolist()
    X_filenames = np.array(X_filenames)[ind].tolist()
    y_filenames = np.array(y_filenames)[ind].tolist()
    return X_filenames, y_filenames


def val_paths(config):
    return common_paths(config.val_X_path, config.val_y_path, config)


def common_paths(x_path, y_path, config):
    file_names = [f for f in sorted(os.listdir(x_path), reverse=True)
                  if np.array([f.endswith(suffix) for suffix in config.suffixs]).any()]

    X_filenames = [os.path.join(x_path, f) for f in file_names]
    y_filenames = []
    for f in file_names:
        for suffix in config.suffixs:
            for possible in ['.', '_ori.']:
                fp_t = os.path.join(y_path, f.split('.')[0] + possible + suffix)
                if os.path.exists(fp_t):
                    y_filenames.append(fp_t)

    assert len(X_filenames) == len(y_filenames)
    return X_filenames, y_filenames


def flush_screen():
    import sys
    sys.stdout.write('\r' + str('-') + ' ' * 20)
    sys.stdout.flush()  # important


def rgb2gray(rgb):
    res = np.dot(rgb[..., :3], [1 / 3., 1 / 3., 1 / 3.])
    res = res[..., np.newaxis]
    return res


class MyConfig(object):
    train_y_path = 'data/voc2012_ori/'
    train_X_path = 'data/voc2012_corr/'
    val_X_path = 'data/val_corr/'
    val_y_path = 'data/val_ori/'
    test_X_path = 'data/test_corr/'
    test_y_path = 'data/test_ori/'
    test_yo_path = 'data/test_restore/'

    suffixs = ['png', 'jpg', 'JPEG']
    train_img_shape = (256, 256)

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
        if 'gray' not in type:
            self.output_channels = 3
        else:
            self.output_channels = 1
        self.type = type
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

        if 'gray' in type:
            self.input_channels = 2
        else:
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
        if img.shape[-1] == 1:
            img = img[..., 0]
            cmap = 'gray'
        multiprocessing.Process(target=my_imshow, args=(img, cmap, True, name)).start()


def my_dbg():
    from IPython import embed;
    embed()


if __name__ == "__main__":
    import time

    config = MyConfig(type="gray_denoise", train_epochs=2, train_batch_size=16)
    print len(train_paths(config)[1])
    print len(val_paths(config)[1])
    for x, y in gen_from_dir(config, mode=True):
        print x.shape, y.shape, time.time()
        xt = x[0][..., :1]
        xt = (xt * 255).astype('uint8')
        yt = y[0][..., :1]
        yt = (yt * 255).astype('uint8')
        my_imshow(xt)
        my_imshow(yt, name='yt')
        break
    my_dbg()
