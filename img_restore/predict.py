import os
import os.path as osp
import pprint
import time

import numpy as np
import pandas as pd
from scipy.misc import imread, imsave


def predict(name='deep_denoise', epochs=0, verbose=2, trainable=True, show=False):
    import keras, utils
    import model as MyModels

    config = utils.MyConfig(type=name, epochs=epochs, batch_size=1024, verbose=verbose)
    model = MyModels.__dict__[name + '_model'](input_shape=(None, None) + (config.input_channels,), trainable=trainable)
    # config2 = utils.MyConfig(type='gray_denoise', epochs=epochs, batch_size=1024, verbose=verbose)  # _wide
    # model2 = MyModels.gray_denoise_model((None, None, 2))
    try:
        model.load_weights(config.model_path, by_name=True)
        # model2.load_weights(config2.model_path, by_name=True)
        pass
    except Exception as inst:
        print inst
        exit(-2)

    # if config.verbose != 2: model2.summary()

    callback_list = [
        keras.callbacks.EarlyStopping(monitor='val_loss2acc', min_delta=0.001, patience=20)
    ]

    x_fns, y_fns = utils.common_paths(config.test_X_path, config.test_y_path, config)
    res = {}
    print 'Prepare OK, timer start'
    for iter_ind, (x_fn, y_fn) in enumerate(zip(x_fns, y_fns)):
        tic = time.time()
        print "Testing ", iter_ind, x_fn, y_fn
        corr_img = imread(x_fn, mode='RGB')
        ori_img = imread(y_fn, mode='RGB')

        # if np.array_equal(corr_img[..., 0], corr_img[..., 1]) or epochs != 0:
        def limits(res):
            res = res.copy()
            # choose_to = min(102400, res.shape[0])
            choose_to=res.shape[0]
            ind = np.random.permutation(res.shape[0])[:choose_to]
            res = res[ind]
            return res, res.copy()
        if np.array_equal(corr_img[..., 0], corr_img[..., 1]):
            continue
            x = utils.img2x(corr_img, config2, patch_size=8)
            assert x.max() < 2.
            x[..., 1] = x[..., 0].copy()
            y = model2.predict(x, batch_size=config.batch_size)
            assert y.max() < 2.
            restore_img = utils.y2img(restore_img=y, corr_img=corr_img, config=config2)
        else:
            # continue
            x = utils.img2x(corr_img, config, patch_size=8)
            x_train, y_train = limits(x)
            model.fit(x_train, y_train, batch_size=config.batch_size * 2,
                      epochs=100,
                      verbose=2, callbacks=callback_list,
                      validation_split=0.1)

            assert x.max() < 2.
            # x[..., 3:] = x[..., 0:3].copy()
            y = model.predict(x, batch_size=config.batch_size)
            assert y.max() < 2.
            restore_img = utils.y2img(restore_img=y, corr_img=corr_img, config=config)

        mse_t = utils.my_mse(ori_img, restore_img)
        toc = time.time()
        res[osp.basename(x_fn)] = (mse_t, toc - tic)
        print 'MSE is', mse_t, 'Take Time', toc - tic
        if restore_img.shape[-1] == 1 and np.array_equal(corr_img[..., 0], corr_img[..., 1]):
            restore_img = restore_img.mean(axis=-1)
        restore_img = restore_img.astype('uint8')
        imsave(osp.join(
            config.test_yo_path,
            osp.basename(x_fn))
            , restore_img)
        if show:
            cmap = 'gray' if len(restore_img.shape) == 2 else None

            # print 'plt start'
            # utils.my_imshow(
            #     (utils.combine_patches(y, corr_img.shape) * 255.).astype('uint8'),
            #     cmap,
            #     block=False,
            #     name='pred'
            # )
            # utils.my_imshow(ori_img, cmap, block=False, name='ori')
            utils.my_imshow(restore_img, cmap, block=False, name='restore')
            # utils.my_imshow(corr_img, cmap, block=False, name='corr')
            # if osp.basename(x_fn) == 'A.png': break
            # break

    return res


def predict_wrapper(queue, name='deep_noise', epochs=0, verbose=2, trainable=True, show=False):
    res = predict(name=name, epochs=epochs, verbose=verbose, trainable=trainable, show=show)
    queue.put(res)


if __name__ == '__main__':
    import argparse


    def parser():
        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='net work compression')
        parser.add_argument('--epoch', dest='epochs', help='number epoch',
                            default=0, type=int)
        parser.add_argument('--verbose', dest='verbose', help="global verbose",
                            default=2, type=int)
        _args = parser.parse_args()
        return _args


    args = parser()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.verbose)
    # # grid search parameter:
    #     queue = multiprocessing.Queue()
    #     rres = []
    #     for args.epochs in [0, 1, 3, 5]:
    #         for trainable in [True, False]:
    #             for name in ['deep_denoise', 'deep_wide_denoise']:
    #                 tic = time.time()
    #                 p = multiprocessing.Process(target=predict_wrapper, kwargs=dict(
    #                     queue=queue,
    #                     name=name,
    #                     epochs=args.epochs,
    #                     verbose=args.verbose,
    #                     trainable=trainable,
    #                     show=False
    #                 ))
    #                 p.start()
    #                 res = queue.get()
    #                 p.join()
    #                 pprint.pprint(res)
    #                 rres.append([args.epochs, name, trainable, res, time.time() - tic])
    #
    #     pprint.pprint(rres)
    args.epochs, trainable, name = 0, False, 'deep_wide_denoise'
    res = predict(name=name, epochs=args.epochs, verbose=2, trainable=trainable, show=True)
    df = pd.DataFrame.from_dict(res).transpose()
    df.columns = ['MSE', 'Time']
    pprint.pprint(df)
