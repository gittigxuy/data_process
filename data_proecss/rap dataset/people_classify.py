# -*- coding:utf-8 -*-
__author__ = 'xuy'
# USAGE
'''
#测试模型的代码
python people_classify.py --model output/shirt_color.model 	 --colorbin output/shirt_color_lb.pickle 	--image /home/xuy/桌面/code/python/caffe_code/bag_Gender_hair_classification/people-det-base/JPEGImages/IMG_020002.jpg


'''


# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2
from PIL import Image
import numpy

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained model model")
# # ap.add_argument("-l", "--categorybin", required=True,
# # 	help="path to output category label binarizer")
# ap.add_argument("-c", "--colorbin", required=True,
# 	help="path to output color label binarizer")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())


#在我这里传入的people_pic是PIL库的,因此需要先转化为opencv
def shirt_color_classify(people_pic):
    # img=Image.open(people_pic)
    image=cv2.cvtColor(numpy.asarray(people_pic),cv2.COLOR_RGB2BGR)

    # load the image
    #我需要先从PIL转化为opencv格式，才可以继续读取
    # image = cv2.imread(people_pic)
    output = imutils.resize(image, width=400)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model_path='/home/xuy/code/keras-yolo3-shirt_color/shirt_color_recognise/output/shirt_color.model'
    pickle_path='/home/xuy/code/keras-yolo3-shirt_color/shirt_color_recognise/output/shirt_color_lb.pickle'

    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network from disk, followed
    # by the category and color label binarizers, respectively
    print("[INFO] loading network...")
    model = load_model(model_path, custom_objects={"tf": tf})
    # categoryLB = pickle.loads(open(args["categorybin"], "rb").read())
    colorLB = pickle.loads(open(pickle_path, "rb").read())

    # classify the input image using Keras' multi-output functionality
    print("[INFO] classifying image...")

    colorProba = model.predict(image)

    # (categoryProba, colorProba) = model.predict(image)

    # find indexes of both the category and color outputs with the
    # largest probabilities, then determine the corresponding class
    # labels
    # categoryIdx = categoryProba[0].argmax()
    colorIdx = colorProba[0].argmax()
    # categoryLabel = categoryLB.classes_[categoryIdx]
    colorLabel = colorLB.classes_[colorIdx]
    color_prob=colorProba[0][colorIdx] * 100
    return (output,colorLabel,color_prob)

#
# output,colorLabel,color_prob=shirt_color_classify('/home/xuy/桌面/code/python/caffe_code/bag_Gender_hair_classification/people-det-base/JPEGImages/IMG_020004.jpg')
# # draw the category label and color label on the image
# # categoryText = "category: {} ({:.2f}%)".format(categoryLabel,
# # 	categoryProba[0][categoryIdx] * 100)
# colorText = "color: {} ({:.2f}%)".format(colorLabel,color_prob)
# # cv2.putText(output, categoryText, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
# # 	0.7, (0, 255, 0), 2)
# cv2.putText(output, colorText, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
#     0.7, (0, 255, 0), 2)
#
# # display the predictions to the terminal as well
# # print("[INFO] {}".format(categoryText))
# print("[INFO] {}".format(colorText))
#
# # show the output image
# cv2.imshow("Output", output)
# cv2.waitKey(0)