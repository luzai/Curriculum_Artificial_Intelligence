from config import Config
import subprocess
import utils, multiprocessing

config = Config('baseline')
from model import LeNet

model = LeNet(config)
model.train()
