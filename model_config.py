
class Config(object):
    def __init__(self, name, nb_conv=2
                 , nb_pool=2, conv_size=(5, 5),
                 nb_conv_chnl=(32, 64), activation='ReLU',
                 fc_size=(512, 128), dp_ratio=.5,
                 nb_fc=2, lr=1E-3):
        self.name = name

        self.nb_conv, \
        self.nb_pool, self.conv_size, \
        self.nb_conv_chnl, self.activation, \
        self.fc_size, self.dp_ratio, \
        self.nb_fc, self.lr \
            = nb_conv, nb_pool, \
              conv_size, nb_conv_chnl, \
              activation, fc_size, \
              dp_ratio, nb_fc, lr
        self.nb_pool = self.nb_conv

        self.make_hparam_str()

    def make_hparam_str(self,hparam=None):
        if hparam is None:
            self.hparam_str = \
                "nbconv={:d}_convsize={}_nbconvchnl={}_act={:s}_fc_size={}_dprat={:.2f}_nbfc={:d}_lr={:.2e}".format(
                    self.nb_conv,  # self.nb_pool,
                    self.conv_size, self.nb_conv_chnl,
                    self.activation,
                    self.fc_size,
                    self.dp_ratio,
                    self.nb_fc,
                    self.lr)
        else:
            self.hparam_str="".format(hparam)
