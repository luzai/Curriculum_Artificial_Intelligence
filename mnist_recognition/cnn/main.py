from config import Config
import subprocess
import utils, multiprocessing

queue = multiprocessing.Queue(10)


def train(queue, config):
    from model import LeNet
    model = LeNet(config)
    model.train()


def run(config):
    t = multiprocessing.Process(target=train, args=(queue, config))
    t.start()
    tasks.append(t)
    t.join()


if __name__ == '__main__':
    subprocess.call('rm -rf log'.split())

    tasks = []
    # a
    # conv=\d,3,.*32, 64.*_act=relu_fc=2,\(512, 128\)_dp=0.50
    for nb_conv_pool, conv_chls in zip([1, 2, 3], [[32], [32, 64], [32, 64, 64]]):
        config = Config('conv{}'.format(nb_conv_pool), nb_conv_pool=nb_conv_pool, conv_chnls=conv_chls)
        run(config)
    # b
    # conv=2,\d,.*32, 64.*_act=relu_fc=2,\(512, 128\)_dp=0.50
    for conv_size in [1, 5, 7]:  # baseline : 3
        config = Config('conv_size{}'.format(conv_size), conv_size=(conv_size, conv_size))
        run(config)
    # c
    # conv=2,3,.*32, \d+.*_act=relu_fc=2,\(512, 128\)_dp=0.50
    for conv_chl in [16, 32, 128]:  # baseline : 64
        config = Config('conv_chnl{}'.format(conv_chl), conv_chnls=(32, conv_chl))
        run(config)
    # d
    # conv=2,3,.*32, 64.*_act=.*_fc=2,\(512, 128\)_dp=0.50
    for act in ['sigmoid', 'tanh']:  # baseline : relu
        config = Config('act_{}'.format(act), act=act)
        run(config)
    # e
    # conv=2,3,.*32, 64.*_act=relu_fc=2,.*_dp=0.50
    for fc_size in [1024, 512, 20]:  # baseline: 128
        config = Config('fc_size{}'.format(fc_size), fc_size=[512, fc_size])
        run(config)
    # f
    # conv=2,3,.*32, 64.*_act=relu_fc=2,\(512, 128\)_dp=.*
    for dprat in [.1, .9]:  # baseline : .9
        config = Config('dp_ratio{}'.format(dprat), dp_rat=dprat)
        run(config)
    # g
    # conv=2,3,.*32, 64.*_act=relu_fc=\d,.*_dp=0.50
    for nb_fc, fc_size in zip([3], [[512, 128, 128]]):  # baseline [2] [[512,128]]
        config = Config('fc_layer{}'.format(nb_fc), nb_fc=nb_fc, fc_size=fc_size)
        run(config)
