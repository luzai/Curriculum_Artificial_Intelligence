import subprocess
from scipy.misc import *
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
import numpy as np
import os

ori_prefix = 'voc2012_ori/'
cmd = 'mkdir -p ' + ori_prefix
subprocess.call(cmd.split())

corr_prefix = 'voc2012_corr/'
cmd = 'mkdir -p ' + corr_prefix
subprocess.call(cmd.split())


def gen_img(img_name):
    print os.path.basename(img_name)
    ori_img_name = ori_prefix + os.path.basename(img_name).split('.')[0] + '.png'
    corr_img_name = corr_prefix + os.path.basename(img_name).split('.')[0] + '.png'

    img_np = imread(img_name, mode='RGB')

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    img_np = rgb2gray(img_np)
    img_np = np.stack([img_np, img_np.copy(), img_np.copy()], axis=-1)

    ori_img = imresize(img_np, (256, 256, 3))
    # ori_img[ori_img == 0] = 1
    # assert ori_img.all(), "All should not be corrupted"
    # print img_np.shape, ori_img.shape

    assert ori_img.dtype == np.uint8
    imsave(ori_img_name, ori_img)

    noise_mask = np.ones(shape=ori_img.shape, dtype=np.uint8)
    rows, cols, chnls = ori_img.shape
    noise_ratio = [0.4, 0.6, 0.8][np.random.randint(low=0, high=3)]
    noise_num = int(noise_ratio * cols)

    # for chnl in range(chnls):
    for row in range(rows):
        choose_col = np.random.permutation(cols)[:noise_num]
        noise_mask[row, choose_col, 0] = 0
    noise_mask = np.stack([noise_mask[..., 0].copy(),
                           noise_mask[..., 0].copy(),
                           noise_mask[..., 0].copy()], axis=-1)

    corr_img = np.multiply(ori_img, noise_mask)
    assert corr_img.dtype == np.uint8

    def summary(x, name='o'):
        pass
        # print name, ' ', x.ravel()[:10]

    imsave(corr_img_name, corr_img)
    ori_img2 = imread(ori_img_name, mode='RGB')
    corr_img2 = imread(corr_img_name, mode='RGB')

    summary(ori_img)
    summary(ori_img2)

    summary(corr_img)
    summary(corr_img2)

    assert np.array_equal(ori_img, ori_img2), 'should equal'
    assert np.array_equal(corr_img, corr_img2), 'same'


img_l = glob.glob('voc2012/*.jpg')
import subprocess
# subprocess.call('rm voc2012_corr -rf'.split())
# subprocess.call('mkdir voc2012_corr -p'.split())
# limits=len(img_l)//5
# limits=32
# img_l = img_l[:limits]
import multiprocessing as mp

pool_size = mp.cpu_count() * 4
pool = mp.Pool(processes=pool_size)
pool.map(gen_img, img_l)
pool.close()
pool.join()

if __name__ == '__main__':
    pass
