from __future__ import division
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
from keras_alfnet import config
from keras_alfnet.model.model_2step import Model_2step

# pass the settings in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# define paths for weight files and detection results
w_path = 'data/models/city_res50_2step.hdf5'
data_path = 'data/examples/'
val_data = os.listdir(data_path)
# out_path = os.path.join(data_path,'detections')
out_path = os.path.join(data_path,'one_pic')
if not os.path.exists(out_path):
    os.makedirs(out_path)

C.random_crop = (1024, 2048)
C.network = 'resnet50'

# C.anchor_box_scales = [[8, 48], [48, 80], [160,200], [200,320]]
# define the ALFNet network
model = Model_2step()
model.initialize(C)
model.creat_model(C, val_data, phase='inference')

img_path='/home/xuy/code/ALFNet/data/cityperson/images/val/lindau/lindau_000024_000019_leftImg8bit.png'
video_path='/home/xuy/code/ALFNet/pedestrain_video_1.mp4'
# video_path='/home/xuy/code/ALFNet/pedestrain_video_1.mp4'
output_path='/home/xuy/code/ALFNet/pedestrain_video_result.mp4'
# model.demo_onepic(C,img_path,w_path,out_path)
# model.demo(C, val_data, w_path, out_path)
model.demo_video(C,video_path,w_path,output_path)
# model.read_video(video_path)