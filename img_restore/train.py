def main(queue, name):
    import keras
    import utils
    import model as MyModels

    assert name + '_model' in MyModels.__dict__.keys()

    config = utils.MyConfig(type=name, train_epochs=1000, train_batch_size=16)
    model = MyModels.__dict__[name + '_model'](input_shape=(256, 256) + (config.input_channels,))

    try:
        model.load_weights(config.model_path, by_name=True)
        pass
    except Exception as inst:
        print inst
        # exit(-2)

    model.summary()

    callback_list = [keras.callbacks.ModelCheckpoint(
        config.model_path,
        monitor='loss2acc', save_best_only=True,
        mode='max', save_weights_only=False),
        keras.callbacks.TensorBoard(log_dir='tf_tmp/')
    ]
    my_metric = lambda x, y: MyModels.loss2acc(x, y, True)
    my_metric.__name__ = 'loss2acc'
    model.compile(optimizer=keras.optimizers.adam(lr=1e-3), loss=['mse'], metrics=[my_metric])
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


import multiprocessing

mp_queue = multiprocessing.Queue()

# main(mp_queue, 'gray_denoise')
main(mp_queue, 'gray_wide_denoise')
