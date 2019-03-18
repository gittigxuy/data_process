# -*- coding:utf-8 -*- 
__author__ = 'xuy'
"""
使用这个代码检查标注框是否有问题
"""
import cv2
import os

img=cv2.imread('/home/xuy/dataset/citypersons/output_FasterRCNN/cityperson_bak/leftImg8bit/val/frankfurt/frankfurt_000001_013496_leftImg8bit.png')


# x_min=231
# y_min=159
# x_max=231+59
# y_max=159+73

x_min=1792
y_min=258
x_max= 1902
y_max=515



i1_pt1=(int(x_min),int(y_min))
i1_pt2=(int(x_max),int(y_max))

cv2.rectangle(img,pt1=i1_pt1,pt2=i1_pt2,thickness=3,color=(255,0,255))
cv2.imshow('Image',img)

cv2.waitKey(0)


