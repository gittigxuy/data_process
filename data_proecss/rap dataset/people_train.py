# USAGE
# python shirt_color_train.py --dataset dataset --model output/fashion.model \
#	--categorybin output/category_lb.pickle --colorbin output/color_lb.pickle
#执行代码：
"""
python people_train.py -m output/shirt_color.model -c output/shirt_color_lb.pickle -p output/color_shirt

vgg:
-m output/shirt_color_vgg.model -c output/shirt_color_lb_vgg.pickle
"""
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.color_net import FashionNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

import xml.etree.ElementTree as ET
from os import getcwd
import os
import shutil

from PIL import Image
import xml.dom.minidom

# ImgPath='people-det-base/my_pic/'
ImgPath='people-det-base/JPEGImages/'



# ImgPath = 'RAP/RAP_dataset/'
# AnnoPath = 'annotations/'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
	help="path to output model")
# ap.add_argument("-l", "--categorybin", required=True,
# 	help="path to output category label binarizer")
ap.add_argument("-c", "--colorbin", required=True,
	help="path to output color label binarizer")
ap.add_argument("-p", "--plot", type=str, default="output",
	help="base filename for generated plots")
args = vars(ap.parse_args())
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 120
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)
color_list=['black','white','red','yellow','blue','green','purpose','brown','gray','orange','multi_color','other']
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
#保存的是图片名字，没有路径
imagePaths = os.listdir(ImgPath)

random.seed(42)
random.shuffle(imagePaths)

# initialize the data, clothing category labels (i.e., shirts, jeans,
# dresses, etc.) along with the color labels (i.e., red, blue, etc.)
#data用来存储图片路径
data = []
#衣服类别暂时不用
# categoryLabels = []
#衣服颜色类别
colorLabels = []

#将color_id转化为颜色类型
def get_colortype(image_id):
	in_file=open('people-det-base/Annotations/%s.xml'%image_id)
	tree=ET.parse(in_file)
	root=tree.getroot()
	for obj in root.iter('object'):
		cls=obj.find('name').text
		if cls=='top':
			color_id=int(obj.find('color').text)
			color_type=color_list[color_id]

			return color_type

#经过循环之后，data_list中的图片数据与颜色一一对应
# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	# print("当前路径是:",imagePath)
	image_path=os.path.join(ImgPath,imagePath)
	image = cv2.imread(image_path)
	image = cv2.resize(image, (96, 96))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = img_to_array(image)
	data.append(image)
	image_pre, ext = os.path.splitext(imagePath)
	# extract the clothing color and category from the path and
	# update the respective lists
	#类别，需要解析xml文件
	color = get_colortype(image_pre)
	if color !=None:
		# categoryLabels.append(cat)
		print("当前路径是：%s，颜色类别是:%s"%(imagePath,color))
		colorLabels.append(color)


# scale the raw pixel intensities to the range [0, 1] and convert to
# a NumPy array
data = np.array(data, dtype="float") / 255.0
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))



# convert the label lists to NumPy arrays prior to binarization
# categoryLabels = np.array(categoryLabels)
# with open('label_color.txt','w')as fd:
# 	for colorlabel in colorLabels:
# 		fd.write(colorlabel+'\n')
# 	print(len(colorLabels))
colorLabels = np.array(colorLabels)

# binarize both sets of labels
print("[INFO] binarizing labels...")
# categoryLB = LabelBinarizer()
colorLB = LabelBinarizer()
# categoryLabels = categoryLB.fit_transform(categoryLabels)
colorLabels = colorLB.fit_transform(colorLabels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, colorLabels,
	test_size=0.2, random_state=42)
(trainX, testX, trainColorY, testColorY) = split

print('颜色的类别是',colorLB.classes_)
# initialize our FashionNet multi-output network
model = FashionNet.build(96, 96,	numColors=len(colorLB.classes_),	finalAct="softmax")

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
	#"category_output": "categorical_crossentropy",
	"color_output": "categorical_crossentropy"
}
lossWeights = { "color_output": 1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])

# train the network to perform multi-output classification
# 这里仅仅输出color类型
H = model.fit(trainX,
	{ "color_output": trainColorY},
	validation_data=(testX,
		{ "color_output": testColorY}),
	epochs=EPOCHS,
	verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the category binarizer to disk
# print("[INFO] serializing category label binarizer...")
# f = open(args["categorybin"], "wb")
# f.write(pickle.dumps(categoryLB))
# f.close()

# save the color binarizer to disk
print("[INFO] serializing color label binarizer...")
f = open(args["colorbin"], "wb")
f.write(pickle.dumps(colorLB))
f.close()

# plot the total loss, category loss, and color loss
lossNames = ["loss",  "color_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()

# save the losses figure
plt.tight_layout()
plt.savefig("{}_losses.png".format(args["plot"]))
plt.close()

# create a new figure for the accuracies
accuracyNames = [ "color_output_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))

# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
	# plot the loss for both the training and validation data
	ax[i].set_title("Accuracy for {}".format(l))
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Accuracy")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()

# save the accuracies figure
plt.tight_layout()
plt.savefig("{}_accs.png".format(args["plot"]))
plt.close()

"""
epoch 27
15s 640us/step - loss: 0.6449 - acc: 0.7839 - val_loss: 0.7237 - val_acc: 0.7630
Epoch 28/50
"""
