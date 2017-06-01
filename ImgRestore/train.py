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
    import model as MyModels

    assert name + '_model' in MyModels.__dict__.keys()

    config = utils.MyConfig(type=name, train_epochs=1000, train_batch_size=16)
    model = MyModels.__dict__[name + '_model'](input_shape=(256, 256) + (config.input_channels,))

    try:
        model.load_weights(config.model_path, by_name=True)
    except Exception as inst:
        print inst

    model.summary()

    callback_list = [keras.callbacks.ModelCheckpoint(
        config.model_path,
        monitor='val_loss2acc', save_best_only=True,
        mode='max', save_weights_only=False)
        # ,
        # keras.callbacks.EarlyStopping(
        #     monitor='val_loss2acc',
        #     min_delta=0.00001, patience=3)
    ]
    my_metric = lambda x, y: MyModels.loss2acc(x, y, True)
    my_metric.__name__ = 'loss2acc'
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss=['mse'], metrics=[my_metric])
    dbg = False
    model.fit_generator(utils.gen_from_dir(config, mode=True),
                        steps_per_epoch=1 if dbg else utils.get_steps(config, train=True),
                        epochs=2 if dbg else config.train_epochs,
                        callbacks=callback_list,
                        validation_steps=utils.get_steps(config, train=False),
                        validation_data=utils.gen_from_dir(config, mode=False)
                        )

    # model.save(config.model_path)
    queue.put({'model_path': config.model_path})

    # del model
    # import gc
    # gc.collect()
    # K.clear_session()
    # tf_graph = tf.get_default_graph()
    # _sess_config = tf.ConfigProto(
    #     allow_soft_placement=True,
    # )
    # _sess_config.gpu_options.allow_growth = True
    # sess = tf.Session(config=_sess_config, graph=tf_graph)
    # K.set_session(sess)
    # # if dbg:
    # #     utils.my_dbg()


import multiprocessing, time

mp_queue = multiprocessing.Queue()
# Todo train
for name in ['deep_wide_denoise']:  # ['single', 'deep', 'denoise', 'deep_denoise']:
    p = multiprocessing.Process(target=main, args=(mp_queue, name))
    print time.time(), '\n'
    p.start()  # non-blocking
    print time.time(), '\n'
    print mp_queue.get()  # blocking
    print time.time(), '\n'
    p.join()
    print time.time(), '\n'
    print '-' * 10
