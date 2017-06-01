import matplotlib
from scipy.misc import *

matplotlib.use("TkAgg")
import numpy as np

from surprise import Reader,Dataset
import surprise
from surprise.dataset import DatasetAutoFolds
from surprise import SVD
from surprise import Dataset
import utils

name = 'deep_denoise'
config = utils.MyConfig(type=name, epochs=30, batch_size=1024, verbose=2)

class MyDataset(DatasetAutoFolds):
    def __init__(self,reader=None,data=None):
        Dataset.__init__(self, reader)
        self.ratings_file = None
        self.n_folds = 5
        self.shuffle = True
        if data.shape[-1]==3:
            data=np.concatenate((data,
                                 np.array([None] * data.shape[0])[..., np.newaxis]),
                                axis=-1)
        self.raw_ratings = data

    def read_ratings(self, file_name):
        if self.raw_ratings is None:
            Dataset.read_ratings(self,file_name)
        else:
            return self.raw_ratings



def restore(x):
    from scipy.sparse import coo_matrix
    sparse_mat = coo_matrix(x)
    data=np.stack([sparse_mat.col,
              sparse_mat.row,
              sparse_mat.data
              ],axis=1).astype('int')
    dataset=MyDataset(data=data)
    trainset=dataset.build_full_trainset()
    algo=SVD()
    algo.train(trainset)

    xx=np.arange(0,x.shape[0])
    yy=np.arange(0,x.shape[1])

    y3,x3=np.meshgrid(yy,xx)
    data=np.stack([x3.ravel(),y3.ravel(),x.ravel()],axis=1).astype('int')
    np.savetxt('tmp.txt',data,fmt='%d')
    data=Dataset.load_from_file('tmp.txt',reader=reader)
    testset=data.construct_testset(data)
    predictions=algo.test(testset)
    return  predictions.reshape(x.shape)


x_fns, y_fns = utils.common_paths(config.test_X_path, config.test_y_path, config)
for iter_ind, (x_fn, y_fn) in enumerate(zip(x_fns, y_fns)):
    print iter_ind, x_fn, y_fn
    corr_img = imread(x_fn, mode='RGB')
    ori_img = imread(y_fn, mode='RGB')
    for chl in range(corr_img.shape[-1]):
        restore(corr_img[..., chl])
