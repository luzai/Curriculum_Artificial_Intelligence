class Config(object):
    def __init__(self, name, nb_conv_pool=2, conv_size=(3, 3),
                 conv_chnls=(32, 64), act='relu',
                 fc_size=(512, 128), dp_rat=.5,
                 nb_fc=2, lr=1E-3):
        self.name = name

        self.nb_conv, \
        self.nb_pool, self.conv_size, \
        self.nb_conv_chnl, self.activation, \
        self.fc_size, self.dp_ratio, \
        self.nb_fc, self.lr \
            = nb_conv_pool, nb_conv_pool, \
              conv_size, conv_chnls, \
              act, fc_size, \
              dp_rat, nb_fc, lr
        self.nb_pool = self.nb_conv

        self.make_hparam_str()

    def make_hparam_str(self, hparam=None):
        if hparam is None:
            self.hparam_str = \
                "conv={:d},{},{}_act={:s}_fc={:d},{}_dp={:.2f}".format(
                    self.nb_conv,  # self.nb_pool,
                    self.conv_size[0], self.nb_conv_chnl,
                    self.activation,
                    self.nb_fc,
                    self.fc_size,
                    self.dp_ratio)
        else:
            self.hparam_str = "".format(hparam)
        # if self.name is not None:
        #     self.hparam_str=self.name
