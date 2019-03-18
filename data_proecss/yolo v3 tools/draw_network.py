# -*- coding:utf-8 -*- 
__author__ = 'xuy'
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from train import *

annotation_path = '/home/xuy/code/keras-yolo3/2007_train.txt'
log_dir = 'logs/'
classes_path = 'model_data/voc_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)


is_tiny_version = len(anchors)==6 # default setting,tiny_anchors==6,anchors==9

if is_tiny_version:
    model = create_tiny_model((416,416), anchors, num_classes,
                              freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
else:
    model = create_model((416,416), anchors, num_classes,
                         freeze_body=2, weights_path='model_data/yolo_weights.h5')  # make sure you know what you freeze
plot_model(model, to_file="my_darknet53.png", show_shapes=True)