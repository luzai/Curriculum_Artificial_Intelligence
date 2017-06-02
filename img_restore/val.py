import multiprocessing
import numpy as np

queue = multiprocessing.Queue()
name = 'gray_denoise'
global x
import keras
import model as MyModels
import utils
from scipy.misc import imread

assert name + '_model' in MyModels.__dict__.keys()

config = utils.MyConfig(type=name, train_epochs=1000, train_batch_size=16)
config.train = False
model = MyModels.__dict__[name + '_model'](input_shape=(None, None) + (config.input_channels,))

try:
    # os.remove(config.model_path)
    model.load_weights(config.model_path, by_name=True)
except Exception as inst:
    print inst
    # exit(-2)
    # os.remove(config.model_path)

model.summary()

callback_list = [keras.callbacks.ModelCheckpoint(
    config.model_path,
    monitor='val_loss2acc', save_best_only=True,
    mode='max', save_weights_only=False),
    keras.callbacks.EarlyStopping(
        monitor='val_loss2acc',
        min_delta=0.1, patience=3)
]
my_metric = lambda x, y: MyModels.loss2acc(x, y, True)
my_metric.__name__ = 'loss2acc'
model.compile(optimizer=keras.optimizers.Adam(lr=1 - 3), loss=['mse'], metrics=[my_metric])
dbg = True
queue.put({'model_path': config.model_path})
if 'gray' in config.type:
    ind = 1
else:
    ind = 3
for x, y in utils.gen_from_dir(config, True):
    break
    y_pred = model.predict(x)
    utils.my_imshow(x[0][..., :ind], block=False)
    utils.my_imshow(y[0][..., :ind], block=False)
    y_pred[0][..., :ind] = utils.post_process(x[0][..., :ind], y_to=y_pred[0][..., :1], config=config)
    utils.my_imshow(y_pred[0][..., :ind], block=False, name='pred_train')
    print utils.my_mse(y_pred[0][..., :ind], x[0][..., :ind])
    break
cnt = 0

tmp = imread('data/val_corr/cara1_04.png', mode='RGB')
tmp = tmp.mean(axis=-1) / 255.
for x, y in utils.gen_from_dir(config, False):
    x_1 = x.copy()
    y_pred = model.predict(x)
    # utils.my_imshow(x[0][..., :ind], block=False)
    # utils.my_imshow(y[0][..., :ind], block=False)
    y_pred[0][..., :ind] = utils.post_process(x[0][..., :ind], y_to=y_pred[0][..., :1], config=config)
    utils.my_imshow(y_pred[0][..., :ind] * 255., block=False, name='pred_val')
    print utils.my_mse(y_pred[0][..., :ind], x[0][..., :ind])
    cnt += 1
    if cnt > 4:
        break

tmp = x[0, :, :, :1].copy()
corr_img = np.concatenate([tmp, tmp.copy(), tmp.copy()], axis=-1)
corr_img = corr_img * 255.

x = utils.img2x(corr_img, config, patch_size=8)
x = np.tile(x, (16, 1, 1, 1))
x_2 = x.copy()
pred = model.predict(x)[0]

restore = utils.y2img(pred, corr_img=corr_img, config=config)

utils.my_imshow(restore, cmap='gray')

from IPython import embed;

embed()
