
# coding: utf-8

# In[255]:

import tensorflow as tf
import numpy as np
print(tf.__path__)
print(tf.__version__)
import keras
print(keras.__version__)
# from keras_frcnn import resnet as nn
from keras.optimizers import Adam, SGD
from keras.layers import Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras_frcnn import losses
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K 
sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        # log_device_placement=True,
        # inter_op_parallelism_threads=8,
        # intra_op_parallelism_threads=8,
#         device_count={'GPU': 0}
        # device_count={'CPU': 0}
    )

sess_config.gpu_options.allow_growth = True
# sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7

sess=tf.Session(config=sess_config)

K.set_session(sess)
import subprocess

proc = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE, shell=True)
(out, err) = proc.communicate()
print ("program output:", out)


# In[256]:

import random
import pprint
import sys
import json
import tensorflow as tf
from keras_frcnn import config

sys.setrecursionlimit(40000)

C = config.Config()
C.num_rois = 2

C.use_vertical_flips = True

from keras_frcnn.pascal_voc_parser import get_data

all_imgs, classes_count, class_mapping = get_data("data")

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

with open('classes.json', 'w') as class_data_json:
    json.dump(class_mapping, class_data_json)

inv_map = {v: k for k, v in class_mapping.iteritems()}

# pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))
random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

from keras_frcnn import data_generators
from keras import backend as K

data_gen_train = data_generators.get_anchor_gt(train_imgs, class_mapping, classes_count, C, K.image_dim_ordering(),
                                               mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, class_mapping, classes_count, C, K.image_dim_ordering(),
                                             mode='val')

from keras_frcnn import resnet as nn
from keras.optimizers import Adam, SGD
from keras.layers import Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_frcnn import losses
from keras.callbacks import ReduceLROnPlateau

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)


img_input = Input(shape=input_shape_img)

roi_input = Input(shape=(C.num_rois, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

# the classifier is build on top of the base layers + the ROI pooling layer + extra layers
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

# define the full model
model = Model([img_input, roi_input], rpn + classifier)

try:
    print 'loading weights from ', C.base_net_weights
    model.load_weights(C.base_net_weights, by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
        'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
        'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    ))

optimizer = Adam(1e-5, decay=0.0)
model.compile(optimizer=optimizer,
              loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), losses.class_loss_cls,
                    losses.class_loss_regr(C.num_rois, len(classes_count) - 1)],
              metrics={'dense_class_{}_loss'.format(len(classes_count)): 'accuracy'})

nb_epochs = 100

callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=0),
             ModelCheckpoint(C.model_path, monitor='val_loss', save_best_only=True, verbose=0),
             ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)]
train_samples_per_epoch = 500  # len(train_imgs)
nb_val_samples = 100  # len(val_imgs),

print 'Starting training'
import os
import subprocess

proc = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE, shell=True)
(out, err) = proc.communicate()
print "program output:", out


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

import os,sys
get_ipython().magic(u'cd /home/xlwang/Faster-RCNN_TF/tools')
# print os.getcwd()
get_ipython().magic(u'cd ../tools/')
import _init_paths

import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
get_ipython().magic(u'cd ..')


# In[ ]:


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


#CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default="./model/VGGnet_fast_rcnn_iter_70000.ckpt")

    args = parser.parse_args()

    return args


# In[258]:

cfg.TEST.HAS_RPN = True  # Use RPN for proposals
print(tf.__version__)
print(tf.__path__)

# init session
sess_config = tf.ConfigProto(
    allow_soft_placement=True,
#     log_device_placement=True,
#     inter_op_parallelism_threads=8,
#     intra_op_parallelism_threads=8,
    device_count = {'GPU': 0}
    # device_count={'CPU': 0}
)

sess_config.gpu_options.allow_growth = True
# sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8

sess = tf.Session(config=sess_config)
# load network
tf.reset_default_graph()
tf.reset_default_graph()
net = get_network("VGGnet_test",21)


# In[260]:

graph=tf.get_default_graph()
graph_def=graph.as_graph_def()
# /print graph_def


# In[261]:

# boilerplate code
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML

import tensorflow as tf


# In[262]:

layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))


# Helper functions for TF Graph visualization

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)
    return strip_def
  
def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def
  
def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

# Visualizing the network graph. Be sure expand the "mixed" nodes to see their 
# internal structure. We are going to visualize "Conv2D" nodes.
tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
show_graph(tmp_def)


# In[263]:

show_graph(graph_def)


# In[264]:

# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations.
# layer = 'mixed4d_3x3_bottleneck_pre_relu'
# channel = 139 # picking some feature channel to visualize

# start with a gray image with a little noise
img_noise = np.random.uniform(size=(224,224,3)) + 100.0

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))
    
def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    
    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input:img})
        # normalizing the gradient, so the same step size should work 
        g /= g.std()+1e-8         # for different layers and networks
        img += g*step
        print(score, end = ' ')
#     clear_output()
    showarray(visstd(img))



# In[265]:

showarray(img_noise/255.)


# In[266]:

img=PIL.Image.open('experiments/flower.jpg')
img=np.float32(img)
showarray(img/255.)


# In[267]:

img=img.reshape((-1,)+img.shape)
img.shape


# In[ ]:




# In[275]:

tf.reset_default_graph()
tf.reset_default_graph()
get_ipython().magic(u'run tools/demo.py')


# In[276]:

input_tensor=graph.get_tensor_by_name("Placeholder:0")
feature_map_tensor=graph.get_tensor_by_name("conv5_3/Conv2D:0")
feature_map=sess.run([feature_map_tensor],{input_tensor:img})


# In[277]:

from pprint import pprint 
len(feature_map),feature_map[0].shape
def m_norm(img0):
    img=img0.copy()
    img+=img.min()
    img/=img.max()
#     pprint(img)
    img*=255
    return img.astype("uint8")


# In[278]:

get_ipython().magic(u'matplotlib inline')
norm_img=m_norm(feature_map[0][0,:,:,:3])
print(norm_img.max(),norm_img.min())
plt.imshow(feature_map[0][0,:,:,0],clim=(-13,0))


# In[279]:

test_img.ravel().shape
test_img.min(),test_img.max()


# In[280]:

test_img=feature_map[0][0,:,:,0]
plt.hist(test_img.ravel(), bins=int(test_img.max()-test_img.min()), range=(test_img.min(),test_img.max()), fc='k', ec='k')
# test_img


# In[281]:

print(img[0].shape)
plt.imshow(img[0][:,:,0],clim=(0,200))


# In[282]:

layers=[op.name for op in graph.get_operations() if op.type=="Placeholder"]
[ graph.get_tensor_by_name(layer+":0") for layer in layers]


# In[283]:

[op.type for op in graph.get_operations() ]


# In[284]:

plt.imshow(img[0])


# In[285]:

import scipy,scipy.io

# load model
saver=tf.train.Saver()
saver.restore(sess, "model/VGGnet_fast_rcnn_iter_70000.ckpt")
# print '\n\nLoaded network {:s}'.format("./model/VGGnet_fast_rcnn_iter_70000.ckpt")


# In[ ]:


# # Warmup on a dummy image
im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
for i in xrange(2):
    _, _= im_detect(sess, net, im)
    os.system("nvidia-smi")
# im=img[0]
from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
# im = imread("data/demo/001763.jpg"
im=plt.imread("data/demo/001763.jpg")
plt.imshow(im)


# In[ ]:




# In[ ]:




# In[ ]:

import time
timer = Timer()
timer.tic()
scores, boxes = im_detect(sess, net, im)
timer.toc()
print(  'Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]) )

# Visualize detections for each class
im = im[:, :, (2, 1, 0)]
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(im, aspect='equal')

CONF_THRESH = 0.5
NMS_THRESH = 0.3
for cls_ind, cls in enumerate(CLASSES[1:]):
    cls_ind += 1 # because we skipped background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
#     print(dets)
    vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)


# In[ ]:

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i])#, interpolation="nearest")#, cmap="gray")
        plt.axis("off")
def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
    plotNNFilter(units)
    return units


# In[ ]:

[n.name for n in tf.get_default_graph().as_graph_def().node]


# In[ ]:

[v.name for v in tf.global_variables()]


# In[ ]:

tf.global_variables()


# In[ ]:

t=tf.get_default_graph().get_operations()[20]
t


# In[ ]:



