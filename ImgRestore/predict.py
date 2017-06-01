import os.path as osp

import numpy as np
from scipy.misc import imread


def predict(name='deep_noise', epochs=0, verbose=2, trainable=True, show=False):
    config = utils.MyConfig(type=name, epochs=epochs, batch_size=1024, verbose=verbose)
    model = MyModels.__dict__[name + '_model'](input_shape=(None, None) + (config.input_channels,), trainable=trainable)
    try:
        model.load_weights(config.model_path, by_name=True)
    except Exception as inst:
        print inst
        exit(-2)

    if config.verbose != 2: model.summary()

    callback_list = [
        keras.callbacks.EarlyStopping(monitor='val_loss2acc', min_delta=0.001, patience=30)
    ]

    x_fns, y_fns = utils.common_paths(config.test_X_path, config.test_y_path, config)
    res = {}
    for iter_ind, (x_fn, y_fn) in enumerate(zip(x_fns, y_fns)):
        if iter_ind > 6:
            break
        print iter_ind, x_fn, y_fn
        # model = utils.shuffle_weights(model)
        corr_img = imread(x_fn, mode='RGB')
        ori_img = imread(y_fn, mode='RGB')
        # if np.array_equal(corr_img[..., 0], corr_img[..., 1]):
        #     continue  # todo

        x = utils.img2x(corr_img, config, patch_size=8)
        assert x.max() < 2.

        def limits(res):
            res = res.copy()
            choose_to = min(102400, res.shape[0])
            ind = np.random.permutation(res.shape[0])[:choose_to]
            res = res[ind]
            return res, res.copy()

        x_train, y_train = limits(x)
        # print  time.time()
        model.fit(x_train, y_train, batch_size=config.batch_size * 2,
                  epochs=config.epochs,  # *2 if iter_ind==0 else config.epochs,
                  verbose=config.verbose, callbacks=callback_list,
                  validation_split=0.1)
        # print  time.time()
        y = model.predict(x, batch_size=config.batch_size)
        assert y.max() < 2.
        # print  time.time()
        restore_img = utils.y2img(restore_img=y, corr_img=corr_img, config=config)

        if show:
            cmap = 'gray' if len(restore_img.shape) == 2 else None

            print 'plt start'
            utils.my_imshow(
                (utils.combine_patches(y, corr_img.shape) * 255.).astype('uint8'),
                cmap,
                block=False,
                name='pred'
            )
            utils.my_imshow(ori_img, cmap, block=False, name='ori')
            utils.my_imshow(restore_img, cmap, block=False, name='restore')
            utils.my_imshow(corr_img, cmap, block=False, name='corr')

        res[osp.basename(x_fn)] = utils.my_mse(ori_img, restore_img)
    return res

def predict_wrapper(queue,name='deep_noise', epochs=0, verbose=2, trainable=True, show=False):
    res=predict(name=name, epochs=args.epochs, verbose=args.verbose, trainable=trainable, show=False)
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

    import os,pprint

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.verbose)
    import keras
    import model as MyModels
    import utils,time,multiprocessing
    queue=multiprocessing.Queue()
    rres=[]
    for args.epochs in [0,1,3,5]:
        for trainable in [True,False]:
            for name in ['deep_denoise','deep_wide_denoise']:
                tic=time.time()
                p=multiprocessing.Process(target=predict_wrapper,kwargs=dict(
                    queue=queue,
                    name=name,
                    epochs=args.epochs,
                    verbose=args.verbose,
                    trainable=trainable,
                    show=False
                ))
                res=queue.get()
                p.join()
                pprint.pprint(res)
                rres.append([args.epochs,name,trainable,res,time.time()-tic])

    pprint.pprint(rres)