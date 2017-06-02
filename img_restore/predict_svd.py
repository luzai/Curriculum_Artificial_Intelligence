import multiprocessing as mp

import matplotlib
from scipy.misc import *

matplotlib.use("TkAgg")
import numpy as np

from surprise import Reader
from surprise.dataset import DatasetAutoFolds
from surprise import SVD
from surprise import Dataset
import utils

name = 'deep_denoise'
config = utils.MyConfig(type=name, epochs=30, batch_size=1024, verbose=2)


class MyDataset(DatasetAutoFolds):
    def __init__(self, reader=None, data=None):
        Dataset.__init__(self, reader)
        self.ratings_file = None
        self.n_folds = 5
        self.shuffle = True
        if data.shape[-1] == 3:
            data = self.change_format(data)
        self.raw_ratings = data

    def read_ratings(self, file_name):
        if self.raw_ratings is None:
            Dataset.read_ratings(self, file_name)
        else:
            return self.raw_ratings

    def change_format(self, data):
        res = []
        for ind in range(data.shape[0]):
            res.append([str(data[ind][0]),
                        str(data[ind][1]),
                        float(data[ind][2]),
                        None])
        return res


def restore(x):
    from scipy.sparse import coo_matrix
    sparse_mat = coo_matrix(x)

    data = np.stack([sparse_mat.col,
                     sparse_mat.row,
                     sparse_mat.data
                     ], axis=1).astype('int')
    np.savetxt('tmp.txt', data, fmt='%d')
    reader = Reader(line_format='user item rating', sep=' ', rating_scale=(0, 255))
    dataset = Dataset.load_from_file('tmp.txt', reader=reader)
    trainset = dataset.build_full_trainset()

    algo = SVD()
    algo.train(trainset)

    xx = np.arange(0, x.shape[0])
    yy = np.arange(0, x.shape[1])

    y3, x3 = np.meshgrid(yy, xx)
    testset = zip(x3.ravel().tolist(), y3.ravel().tolist())
    testset = [str(a) + ' ' + str(b) for (a, b) in testset]
    print testset[:10]

    # pool = mp.Pool(mp.cpu_count() * 2)

    def my_predict(test):
        a, b = test.split()
        return algo.predict(uid=a, iid=b)

    predictions = []
    for test in testset:
        predictions.append(int(my_predict(test).est))
    # predictions=pool.map(my_predict, testset)

    # pool.close()
    # pool.join()
    # print predictions[:10]

    return np.array(predictions).reshape(x.shape)


import os.path as osp

x_fns, y_fns = utils.common_paths(config.test_X_path, config.test_y_path, config)
for iter_ind, (x_fn, y_fn) in enumerate(zip(x_fns, y_fns)):
    print iter_ind, x_fn, y_fn
    corr_img = imread(x_fn, mode='RGB')
    ori_img = imread(y_fn, mode='RGB')
    img_l = []
    for chl in range(corr_img.shape[-1]):
        img_l.append(restore(corr_img[..., chl]))
    img = np.stack(img_l, -1)
    print img.shape
    utils.my_imshow(img, name=osp.basename(x_fn))
