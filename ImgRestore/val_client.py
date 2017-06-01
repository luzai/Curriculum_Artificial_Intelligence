def main(queue, name):
    import keras
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.misc import imread
    import tensorflow as tf
    import model as MyModels
    import utils, os, multiprocessing
    import time
    import keras.backend as K

    assert name + '_model' in MyModels.__dict__.keys()

    config = utils.MyConfig(type=name, train_epochs=1000, train_batch_size=16)
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
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss=['mse'], metrics=[my_metric])
    dbg = True
    # model.fit_generator(utils.gen_from_dir(config, mode=True),
    #                     steps_per_epoch=1 if dbg else utils.get_steps(config, train=True),
    #                     epochs=2 if dbg else config.train_epochs,
    #                     callbacks=callback_list,
    #                     validation_steps=utils.get_steps(config, train=False),
    #                     validation_data=utils.gen_from_dir(config, mode=False)
    #                     )

    # model.save(config.model_path)
    queue.put({'model_path': config.model_path})
    for x, y in utils.gen_from_dir(config, True):
        y_pred = model.predict(x)
        utils.my_imshow(x[0][..., :3], block=False)
        utils.my_imshow(y[0][..., :3], block=False)
        y_pred[0][..., :3] = utils.post_process(x[0][..., :3], y_to=y_pred[0][..., :3])
        utils.my_imshow(y_pred[0][..., :3], block=False, name='pred_train')
        print utils.my_mse(y_pred[0][..., :3], x[0][..., :3])

        break

    for x, y in utils.gen_from_dir(config, False):
        y_pred = model.predict(x)
        utils.my_imshow(x[0][..., :3], block=False)
        utils.my_imshow(y[0][..., :3], block=False)
        y_pred[0][..., :3] = utils.post_process(x[0][..., :3], y_to=y_pred[0][..., :3])
        utils.my_imshow(y_pred[0][..., :3], block=False, name='pred_val')
        print utils.my_mse(y_pred[0][..., :3], x[0][..., :3])
        break

