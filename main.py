import  tensorflow as tf
# tf_graph = tf.get_default_graph()
# _sess_config = tf.ConfigProto(allow_soft_placement=True)
# _sess_config.gpu_options.allow_growth = True
from model import  LeNet
from model_config import Config
import  subprocess

if __name__ == '__main__':
    subprocess.call('rm -rf log'.split())

    config=Config('baseline')
    model=LeNet(config)
    model.train()


