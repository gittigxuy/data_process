# -*- coding:utf-8 -*- 
__author__ = 'xuy'

gender_list=['male','female']
hair_list=['long','short','other']
top_list=['T-shirt','skirt','waitao','rurongfu','xifu','other']#上衣
down_list=['changku','duanku','changqun','duanqun','other']
shoes_list=['pixie','yundongxie','liangxie','xuezi','other']
bag_list=['danjianbao','shuangjianbao','shoulaxiang','qianbao','other']
color_list=['black','white','red','yellow','blue','green','purpose','brown','gray','orange','multi_color','other']

import xml.etree.ElementTree as ET
from os import getcwd
import os
import shutil

ImgPath='people-det-base/JPEGImages/'

def convert_annotation(image_id,list_file):
    in_file=open('people-det-base/Annotations/%s.xml'%image_id)
    tree = ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        cls=obj.find('name').text
        if cls=='top':
            color_id=int(obj.find('color').text)
            xmlbox=obj.find('bndbox')
            b=(int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            #[xmin,ymin,xmax,ymax,color_id]
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(color_id))


imagelist=os.listdir(ImgPath)
list_file=open('people-det-base/people_train.txt','w')

#遍历每个图片名字
for image in imagelist:
    image_pre,ext=os.path.splitext(image)
    list_file.write('people-det-base/JPEGImages/%s.jpg'%(image_pre))
    convert_annotation(image_pre,list_file)
    list_file.write('\n')

list_file.close()

