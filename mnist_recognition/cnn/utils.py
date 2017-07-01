import random, string
import numpy as np
import os
import os.path as osp

root_dir = osp.normpath(osp.join(osp.dirname(__file__)))
os.chdir(root_dir)


def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))


class Dataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.ind = 0

    def next_batch(self, batch_size=1024):
        x = self.images[self.ind:self.ind + batch_size]
        y = self.labels[self.ind:self.ind + batch_size]
        self.ind += batch_size
        self.ind %= self.images.shape[0]
        return x, y


class SVHN:
    def __init__(self):
        (tx, ty), (test_x, test_y) = load_data_svhn()
        print (tx.shape, ty.shape, test_x.shape, test_y.shape)
        self.train = Dataset(tx, ty)
        self.test = Dataset(test_x, test_y)


def load_data_svhn():
    import scipy.io as sio
    import os.path
    import commands

    if os.path.isdir('data/SVHN') == False:
        os.mkdir('data/SVHN')

    data_set = []
    if os.path.isfile('data/SVHN/train_32x32.mat') == False:
        data_set.append("train")
    if os.path.isfile('data/SVHN/test_32x32.mat') == False:
        data_set.append("test")

    try:
        import requests
        from tqdm import tqdm
    except:
        # use pip to install these packages:
        # pip install tqdm
        # pip install requests
        print('please install requests and tqdm package first.')

    for set in data_set:
        print ('download SVHN ' + set + ' data, Please wait.')
        url = "http://ufldl.stanford.edu/housenumbers/" + set + "_32x32.mat"
        response = requests.get(url, stream=True)
        with open("data/SVHN/" + set + "_32x32.mat", "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)

    train_data = sio.loadmat('data/SVHN/train_32x32.mat')
    train_x = train_data['X']
    train_y = train_data['y']

    test_data = sio.loadmat('data/SVHN/test_32x32.mat')
    test_x = test_data['X']
    test_y = test_data['y']

    # 1 - 10 to 0 - 9
    train_y = train_y - 1
    test_y = test_y - 1
    train_y = to_category(train_y)
    test_y = to_category(test_y)

    train_x = np.transpose(train_x, (3, 0, 1, 2))
    test_x = np.transpose(test_x, (3, 0, 1, 2))

    return (train_x, train_y), (test_x, test_y)


def to_category(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
