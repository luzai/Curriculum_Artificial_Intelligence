import keras
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import model as MyModels
import utils,os , multiprocessing
import time

config=utils.MyConfig(type='single',epochs=1000,batch_size=16)
model = MyModels.single_model(input_shape=(None,None,6), kernel_size=20)

# config=utils.MyConfig(type='deep',epochs=1000,batch_size=16,input_channels=6)
# model = MyModels.deep_model(input_shape=(None, None) + (config.input_channels,))

# config = utils.MyConfig(type="deep_denoise", epochs=1000, batch_size=16)
# model = MyModels.deep_denoise_model(input_shape=(None, None, 6),n1=16,n2=128,n3=256)



try:
    # os.remove(config.model_path)
    model.load_weights(config.model_path, by_name=True)

except Exception as inst:
    print inst
    exit(-2)
    # os.remove(config.model_path)


model.summary()

callback_list = [keras.callbacks.ModelCheckpoint(
    config.model_path,
    monitor='val_loss2acc', save_best_only=True,
    mode='max', save_weights_only=False),
    keras.callbacks.EarlyStopping(monitor='val_loss2acc', min_delta=0.1, patience=2)
]

# model.fit_generator(utils.gen_from_dir(config, mode=True),
#                     steps_per_epoch=utils.get_steps(config,train=True),
#                     epochs=config.epochs, callbacks=callback_list,
#                     validation_steps=utils.get_steps(config, train=False),
#                     validation_data=utils.gen_from_dir(config, mode=False)
#                     )

train = False
if not train:
    del callback_list[0]
    x_fns, y_fns = utils.common_paths(config.test_X_path, config.test_y_path, config)
    for x_fn, y_fn in zip(x_fns, y_fns):
        # model=utils.shuffle_weights(model)
        corr_img = imread(x_fn, mode='RGB')
        ori_img = imread(y_fn, mode='RGB')
        if np.array_equal(corr_img[...,0],corr_img[...,1]):
            continue
        mask = utils.get_mask(corr_img)

        x = np.concatenate((utils.img2x(corr_img), mask), axis=2)
        x = utils.make_patches(x, patch_size=8)

        y_true = x.copy()

        model.fit(x, y_true, batch_size=1024*4, epochs=250 if not train else 250,
                  verbose=2, callbacks=callback_list,
                  validation_split=0.1)

        y_out = model.predict(x)
        y_out = utils.combine_patches(y_out, out_shape=corr_img.shape)
        y_out = utils.y2img(y_out)

        restore_img = np.clip(y_out, 0, 255).astype('uint8')
        restore_img = utils.post_process(corr_img, restore_img)

        plt.figure()
        plt.imshow(restore_img, cmap='gray' if len(restore_img.shape) == 2 else None)

        print(utils.mse(ori_img, restore_img))
        # plt.show()
        # import thread
        # thread.start_new_thread(plt.show,())
    plt.show()
else:

    for corr_img, ori_img in utils.gen_from_dir(config, mode=True):
        corr_img = utils.y2img(corr_img)
        ori_img = utils.y2img(ori_img)
        mask = utils.get_mask(corr_img)

        x = np.concatenate((utils.img2x(corr_img), mask), axis=2)
        x = utils.make_patches(x, patch_size=8)

        y_true = x.copy()

        model.fit(x, y_true, batch_size=1024, epochs=10 if not train else 250,
                  verbose=2, callbacks=callback_list,
                  validation_split=0.1)

        y_out = model.predict(x)
        y_out = utils.combine_patches(y_out, out_shape=corr_img.shape)
        y_out = utils.y2img(y_out)

        restore_img = np.clip(y_out, 0, 255).astype('uint8')
        restore_img = utils.post_process(corr_img, restore_img)


        plt.figure()
        plt.imshow(restore_img, cmap='gray' if len(restore_img.shape) == 2 else None)

        print(utils.mse(ori_img, restore_img))
    # model.save(config.model_path)
    plt.show()

plt.show()