# -*- coding:utf-8 -*- 
__author__ = 'xuy'

# coding=utf-8
import scipy.io
import os

#reference:https://blog.csdn.net/qq_33614902/article/details/83313898
# Function：将训练集的annotations转换为YOLOv3训练所需的label/train/XXX.txt格式
# How to run? ###python citypersons2yolo.py
# def convert(size, box):
#     dw = 1. / size[0]
#     dh = 1. / size[1]
#     x = (box[0] + box[1]) / 2.0
#     y = (box[2] + box[3]) / 2.0
#     w = box[1] - box[0]
#     h = box[3] - box[2]
#     x = x * dw
#     w = w * dw
#     y = y * dh
#     h = h * dh
#     return (x, y, w, h)


# You can download anno_train.mat from "https://bitbucket.org/shanshanzhang/citypersons/src/f44d4e585d51d0c3fd7992c8fb913515b26d4b5a/annotations/".
data = scipy.io.loadmat('/home/xuy/dataset/citypersons/shanshanzhang-citypersons/annotations/anno_val.mat')#fix
pic_train='/home/xuy/dataset/citypersons/leftImg8bit/val'#fix
data = data['anno_val_aligned'][0]

# if not os.path.exists('/home/xuy/dataset/citypersons/train_result_txt/'):#fix
#     os.makedirs('/home/xuy/dataset/citypersons/train_result_txt/')#fix
txt_name = '/home/xuy/dataset/citypersons/result_csv_reploss/val.txt'#fix
f = open(txt_name, 'w')
for record in data:
    im_name = record['im_name'][0][0][0]
    bboxes = record['bbs'][0][0]
    (shot_name, extension) = os.path.splitext(im_name)
    # txt_name = os.path.join('/home/xuy/dataset/citypersons/train_result_txt', shot_name + '.txt')#fix

    pic_filename=os.path.join(pic_train,im_name.split('_')[0],im_name)

    # im_name = os.path.join('train', im_name.split('_', 1)[0], im_name)

    for bbox in bboxes:
        class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis = bbox
        #ignore label

        # b = (int(x1), int(int(x1) + int(w)), int(y1), int(int(y1) + int(h)))  # (xmin, xmax, ymin, ymax)
        # b_vis=(int(x1_vis),int(int(x1_vis)+int(w_vis)),int(y1_vis),int(int(y1_vis)+int(h_vis)))#(xmin_vis, xmax_vis, ymin, ymax)

        b = (int(x1),  int(y1),int(int(x1) + int(w)), int(int(y1) + int(h)))  # (xmin, ymin,xmax,  ymax)
        # b_vis = (int(x1_vis), int(y1_vis),int(int(x1_vis) + int(w_vis)) ,int(int(y1_vis) + int(h_vis)))  # (xmin_vis,ymin_vis, xmax_vis,  ymax)
        # bb = convert((int(2048), int(1024)), b)
        # bb_vis = convert((int(2048), int(1024)), b_vis)
        #ignore label
        if class_label == 0:

            f.write(pic_filename+' ' + ' '.join([str(a) for a in b])+' ignore' + '\n')
        else:
            # f.write(pic_filename+' '+' '.join([str(a) for a in b])+' '+ ' '.join([str(a_vis)for a_vis in b_vis])+' person'+'\n')
            f.write(pic_filename+' '+' '.join([str(a) for a in b])+' person'+'\n')

f.close()

#将txt文件转化为csv文件
import csv
with open('/home/xuy/dataset/citypersons/result_csv_reploss/val.csv', 'w+',newline='') as csvfile:#fix
    spamwriter = csv.writer(csvfile, dialect='excel')
    # 读要转换的txt文件，文件每行各词间以@@@字符分隔
    with open('/home/xuy/dataset/citypersons/result_csv_reploss/val.txt', 'r',encoding='utf-8') as filein:#fix
        for line in filein:
            line_list = line.strip('\n').split('\t')
            spamwriter.writerow(line_list)
