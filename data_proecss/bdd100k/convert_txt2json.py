# -*- coding:utf-8 -*- 
__author__ = 'xuy'

'''
将之前通过val生成的txt文件转化为json文件

先遍历txt文件夹的所有文件，改为prefix.jpg的形式当作filename
'''

import json
import os
def iterbrowse(path):
    for home, dirs, files in os.walk(path):
        for filename in files:
            yield os.path.join(home, filename)
def json2txt(txt_filepath):
    #读取每个txt文件的全路径
    boxes = []
    for txt_path in iterbrowse(txt_filepath):
        txt_basename=os.path.basename(txt_path)
        portion = os.path.splitext(txt_basename)
        if portion[1] == '.txt':
            pic_filename = portion[0] + '.jpg'
        txt_file=open(txt_path)
        lines=txt_file.readlines()

        for line in lines:
            line=line.split()
            class_label, prob, xmin, ymin, xmax, ymax = line[-6], line[-5], line[-4], line[-3], line[-2], line[-1]
            if class_label=='sign':
                class_label='traffic sign'
            if class_label=='light':
                class_label = 'traffic light'

            box = {
                'name': pic_filename,
                'timestamp': 1000,
                'category': class_label,
                'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)],
                'score': float(prob)

            }
            boxes.append(box)

    return boxes

#获取txt文件的路径
det=json2txt('txt文件路径')
#生成预测json文件
json.dump(det,open('生成json文件路径','w'),indent=4,separators=(',', ': '))


"""

"""


#
# def json2txt(txt_filename,json_filename=None):
#     #传入的是txt的全路径，需要将其改为basename
#     txt_basename=os.path.basename(txt_filename)
#     portion=os.path.splitext(txt_basename)
#     if portion[1]=='.txt':
#         pic_filename=portion[0]+'.jpg'
#
#     txt_file=open(txt_filename,'r')
#     lines=txt_file.readlines()
#     boxes=[]
#     for line in lines:
#         line=line.split()
#
#         class_label,prob,xmin,ymin,xmax,ymax=line[-6],line[-5],line[-4],line[-3],line[-2],line[-1]
#         box={
#             'name':pic_filename,
#             'timestamp':1000,
#             'category':class_label,
#             'bbox':[xmin,ymin,xmax,ymax],
#             'score':float(prob)
#
#         }
#         boxes.append(box)
#     return boxes
#
# boxes=json2txt('/home/xuy/code/mAP/predicted/b1c9c847-3bda4659.txt')
# print(boxes)







