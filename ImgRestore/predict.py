import glob

import matplotlib.pyplot as plt
import numpy as np

import utils
from model import single_model
from utils import MyConfig

config = MyConfig()
model = single_model(input_shape=(None, None) + (config.input_channels,), kernel_size=6)

model.load_weights(config.sav_path)
model.summary()

from utils import img2x, gray2rgb

img_l = glob.glob('data/test/*')

for img_name in img_l:

    img = plt.imread(img_name)
    if not img.shape[-1] == 3:
        img = gray2rgb(img)
    img = utils.denorm_img(img)
    assert img.dtype == np.uint8
    plt.figure()
    plt.imshow(img)

    assert img.shape[-1] == 3
    x = img2x(img)
    x = utils.norm_img(x)
    y = model.predict(x[np.newaxis, ...])[0]
    y = utils.denorm_img(y)
    assert y.dtype == np.uint8
    y = utils.conserve_img(img, y)

    print utils.mse(img, y)

    plt.figure()
    plt.imshow(y)
plt.show()
