def main(queue, name):
    import keras
    import model as MyModels
    import utils

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
    model.compile(optimizer=keras.optimizers.Adam(lr=1-3), loss=['mse'], metrics=[my_metric])
    dbg = True
    queue.put({'model_path': config.model_path})
    if 'gray' in config.type:
        ind=1
    else:
        ind=3
    for x, y in utils.gen_from_dir(config, True):
        break
        y_pred = model.predict(x)
        utils.my_imshow(x[0][..., :ind], block=False)
        utils.my_imshow(y[0][..., :ind], block=False)
        y_pred[0][..., :ind] = utils.post_process(x[0][..., :ind], y_to=y_pred[0][..., :1], config=config)
        utils.my_imshow(y_pred[0][..., :ind], block=False, name='pred_train')
        print utils.my_mse(y_pred[0][..., :ind], x[0][..., :ind])
        break
    cnt=0
    for x, y in utils.gen_from_dir(config, False):
        y_pred = model.predict(x)
        # utils.my_imshow(x[0][..., :ind], block=False)
        # utils.my_imshow(y[0][..., :ind], block=False)
        y_pred[0][..., :ind] = utils.post_process(x[0][..., :ind], y_to=y_pred[0][..., :1], config=config)
        utils.my_imshow(y_pred[0][..., :ind], block=False, name='pred_val')
        print utils.my_mse(y_pred[0][..., :ind], x[0][..., :ind])
        cnt+=1
        if cnt >30:
            break
#