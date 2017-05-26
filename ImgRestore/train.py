import glob
import keras
import matplotlib.pyplot as plt
import numpy as np
import os

from model import single_model
from utils import MyConfig

import utils

import model

# model = single_model(input_shape=(None, None) + (config.input_channels,), kernel_size=20)
# model = model.deep_model(input_shape=(None, None) + (config.input_channels,))
model = model.deep_denoise_model(input_shape=(None,None,3))
config = utils.MyConfig(type="deep_denoise", epochs=1000,batch_size=16)

try:
    model.load_weights(config.model_path, by_name=True)
except Exception as inst:
    print inst
    os.remove(config.model_path)

model.summary()

callback_list = [keras.callbacks.ModelCheckpoint(
    config.model_path,
    monitor='val_loss2acc', save_best_only=True,
    mode='max', save_weights_only=False)]

model.fit_generator(utils.gen_from_dir(config, mode=True),
                    steps_per_epoch=utils.get_steps(config,train=True),
                    epochs=config.epochs, callbacks=callback_list,
                    validation_steps=utils.get_steps(config, train=False),
                    validation_data=utils.gen_from_dir(config, mode=False)
                    )
model.save(config.model_path)
