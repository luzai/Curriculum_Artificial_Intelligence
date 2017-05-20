import subprocess

import matplotlib

matplotlib.use('TkAgg')
import cv2
import glob
import numpy as np
import os

ori_prefix = 'voc2012_ori/'
cmd = 'mkdir -p ' + ori_prefix
subprocess.call(cmd.split())

corr_prefix = 'voc2012_corr/'
cmd = 'mkdir -p ' + corr_prefix
subprocess.call(cmd.split())

img_l = glob.glob('voc2012/*.jpg')
dbg=False
for ind, img_name in enumerate(img_l):
    if dbg and ind > 10:
        break
    print os.path.basename(img_name)
    ori_img_name = ori_prefix + os.path.basename(img_name)
    corr_img_name = corr_prefix + os.path.basename(img_name)
    img_np = cv2.imread(img_name)
    from scipy.misc import imresize

    ori_img = imresize(img_np, (512, 512, 3))
    ori_img[ori_img==0]=1
    assert ori_img.all(), "All should not be corrupted"
    print img_np.shape, ori_img.shape

    if dbg:
        cv2.imshow('ca',ori_img)
        cv2.waitKey(1000)
    assert ori_img.dtype==np.uint8
    cv2.imwrite(ori_img_name, ori_img)

    noise_mask = np.ones(shape=ori_img.shape, dtype=np.uint8)
    rows, cols, chnls = ori_img.shape
    noise_ratio = [0.4, 0.6, 0.8][np.random.randint(low=0, high=3)]
    noise_num = int(noise_ratio * cols)

    for chnl in range(chnls):
        for row in range(rows):
            choose_col = np.random.permutation(cols)[:noise_num]
            noise_mask[row, choose_col, chnl] = 0
    corr_img = np.multiply(ori_img, noise_mask)
    assert corr_img.dtype==np.uint8
    cv2.imwrite(corr_img_name, corr_img)

# plt.plot(1)
# plt.show()
