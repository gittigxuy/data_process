# -*- coding:utf-8 -*- 
__author__ = 'xuy'

import numpy as np
import json
import os

from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import time
import os
import copy
import argparse
import collections
import sys
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
import cv2
import skimage
"""
最终版本的demo代码，适用于bi-box以及reploss
"""
label_names = ['pedestrian']

def py_cpu_nms(dets, scores, thresh=0.3):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep

def generate_txt_result(img, boxes,labels, waitkey_value=0,img_file="",output_path=""):
    cvimg = np.array(img)
    # if cvimg.channel!='RGB':
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
    isOutput = True if output_path != "" else False
    # pic_result_path='/home/xuy/dataset/citypersons/result_pic/'
    txt_save_path = '/home/xuy/code/Repulsion_Loss/predict_result/annos/'
    for i in range(boxes.shape[0]):
        cv2.rectangle(cvimg, (int(boxes[i, 0]), int(boxes[i, 1])), (int(boxes[i, 2]), int(boxes[i, 3])), (0, 255, 0), 1)
        cv2.putText(cvimg, label_names[labels[i]], (int(boxes[i, 0]), int(boxes[i, 1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 0, 255), 1)
    # show detection
    # cv2.imshow('Detection', cvimg)
    if isOutput:
        # 保存图片结果
        # img_file=output_path+img_file.split('/')[-1]
        # cv2.imwrite(filename=img_file,img=cvimg)
        # 保存txt结果
        save_txt_path = txt_save_path + os.path.basename(img_file)[:-3] + 'txt'
        with open(save_txt_path, 'w', encoding='utf-8') as f:
            for i in range(boxes.shape[0]):
                f.write("%s %d %d %d %d\n"%(label_names[labels[i]],int(boxes[i, 0]),int(boxes[i, 1]),int(boxes[i, 2]),int(boxes[i, 3])))

    key = cv2.waitKey(waitkey_value)
    return cvimg, key

# def draw_box(img, boxes,labels, waitkey_value=0,img_file="",output_path=""):
#img_file表示保存图片的文件名，output_path表示图片保存路径
def draw_box(img, boxes,labels, waitkey_value=0,img_file="",output_path=""):
    cvimg = np.array(img)
    # if cvimg.channel!='RGB':
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
    isOutput = True if output_path != "" else False
    # pic_result_path='/home/xuy/dataset/citypersons/result_pic/'
    for i in range(boxes.shape[0]):
        cv2.rectangle(cvimg,(int(boxes[i,0]),int(boxes[i,1])),(int(boxes[i,2]),int(boxes[i,3])),(0,255,0),2 )
        cv2.putText(cvimg,label_names[labels[i]],(int(boxes[i,0]),int(boxes[i,1])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    #show detection
    # cv2.imshow('Detection', cvimg)
    if isOutput:
        #保存图片结果
        img_file=output_path+img_file.split('/')[-1]
        cv2.imwrite(filename=img_file,img=cvimg)


    key = cv2.waitKey(waitkey_value)
    return cvimg,key

#深度遍历每个文件夹里面所有的文件
def list_all_files(rootdir):
    import os
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files

base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.456), (0.229,0.224,0.225))
        ])
def video_test(video_path,retinanet,iou_threshold=0.3,score_threshold=0.5,max_detections=200,output_path=""):
    retinanet.eval()
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (2048, 1024))

    frame=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    size=frame.size
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    isOutput = True if output_path != "" else False
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")
    data_frame = base_transform(frame)

    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     # print(frame.)
    #     img=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    #     cv2.imshow('ped_detection',frame)
    #     if cv2.waitKey(1)&0xFF==ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    if isOutput:
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        #第三个参数１５表示生成的视频文件的fps
        out = cv2.VideoWriter(output_path, codec,15.0, size)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (2048, 1024))

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        data_frame = base_transform(frame)
        with torch.no_grad():
            scores, labels, boxes = retinanet(data_frame.cuda().float().unsqueeze(dim=0))

            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                boxes = boxes.cpu().numpy()

                # add nms
                index_cls = py_cpu_nms(boxes, scores, thresh=iou_threshold)
                boxes = boxes[index_cls]
                scores = scores[index_cls]
                labels = labels[index_cls]

                # print(boxes,scores,labels)
                # print("############################")

                # add Ot
                # select indices which have a score above the threshold,Ot
                indices = np.where(scores > score_threshold)[0]

                # select those scores，根据indices过滤掉score值小于阈值的index
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections,返回score个box
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
            # 没有检测出行人
            else:
                image_boxes = np.zeros((0, 4))
                image_scores = 0.0
                image_labels = None

        # print(image_scores, image_labels, image_boxes)
            # draw_box(img, boxes, labels)
        #默认waitkey_value＝０如果变成１的话，那么就是不等待
        cvimg,key=draw_box(frame, image_boxes, image_labels,waitkey_value=1)
        cvimg=np.asarray(cvimg)
        if isOutput:
            out.write(cvimg)
        if key==27:
            break
    cap.release()
    cv2.destroyAllWindows()




#测试单张图片
def test_img(img_file,retinanet,pic_output_path="",iou_threshold=0.5,score_threshold=0.5,max_detections=100):
    #input data

    img=cv2.imread(img_file)

    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = cv2.resize(img,(2048, 1024))

    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)


    # height,width,_=img.shape
    # img=img.astype(np.float32) / 255.0
    # base_transform=transforms.Compose([
    # Resizer(), Normalizer()
    # ])

    data_img=base_transform(img)
    # print(data_img.shape)
    retinanet.eval()
    with torch.no_grad():
        scores, labels, boxes = retinanet(data_img.cuda().float().unsqueeze(dim=0))

        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # add nms
            index_cls = py_cpu_nms(boxes, scores, thresh=iou_threshold)
            boxes = boxes[index_cls]
            scores = scores[index_cls]
            labels = labels[index_cls]

            # print(boxes,scores,labels)
            # print("############################")

            # add Ot
            # select indices which have a score above the threshold,Ot
            indices = np.where(scores > score_threshold)[0]

            # select those scores，根据indices过滤掉score值小于阈值的index
            scores = scores[indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            # select detections,返回score个box
            image_boxes = boxes[indices[scores_sort], :]
            image_scores = scores[scores_sort]
            image_labels = labels[indices[scores_sort]]
        #没有检测出行人
        else:
            image_boxes=np.zeros((0,4))
            image_scores=0.0
            image_labels=None
        # print(image_scores, image_labels, image_boxes)
        # draw_box(img, boxes, labels)


        #保存图片,fix
        # draw_box(img, image_boxes, image_labels,waitkey_value=0,img_file=img_file,output_path=pic_output_path)
        #保存txt结果,fix
        generate_txt_result(img, image_boxes, image_labels,waitkey_value=0,img_file=img_file,output_path=pic_output_path)
        #不保存图片,fix
        # draw_box(img, image_boxes, image_labels,waitkey_value=0,img_file=img_file,output_path='')

def generate_single_txt_result_bak(img_path,retinanet,iou_threshold=0.3,score_threshold=0.5,max_detections=100):
    """
    输入是一个图片文件夹，输出是val.txt这个预测结果，然后再通过generate_citypersons_jsonResult.py文件转成可以evaluate的文件

    :param img_path:
    :param retinanet:

    :param iou_threshold:
    :param score_threshold:
    :param max_detections:
    :return:
    """
    filenames=list_all_files(img_path)
    generate_txt_filename='predict_result/val.txt'
    with open(generate_txt_filename, 'w', encoding='utf-8') as f:
        #预测所有的图片
        for filename in filenames:
            # print (type(filename))
            img=cv2.imread(filename)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            img = cv2.resize(img,(2048, 1024))

            if len(img.shape) == 2:
                img = skimage.color.gray2rgb(img)


            # height,width,_=img.shape
            # img=img.astype(np.float32) / 255.0
            # base_transform=transforms.Compose([
            # Resizer(), Normalizer()
            # ])

            data_img=base_transform(img)
            # print(data_img.shape)
            retinanet.eval()
            #开始进行测试
            with torch.no_grad():
                scores, labels, boxes = retinanet(data_img.cuda().float().unsqueeze(dim=0))

                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                    labels = labels.cpu().numpy()
                    boxes = boxes.cpu().numpy()

                    # add nms
                    index_cls = py_cpu_nms(boxes, scores, thresh=iou_threshold)
                    boxes = boxes[index_cls]
                    scores = scores[index_cls]
                    labels = labels[index_cls]

                    # print(boxes,scores,labels)
                    # print("############################")

                    # add Ot
                    # select indices which have a score above the threshold,Ot
                    indices = np.where(scores > score_threshold)[0]

                    # select those scores，根据indices过滤掉score值小于阈值的index
                    scores = scores[indices]

                    # find the order with which to sort the scores
                    scores_sort = np.argsort(-scores)[:max_detections]

                    # select detections,返回score个box
                    image_boxes = boxes[indices[scores_sort], :]
                    image_scores = scores[scores_sort]
                    image_labels = labels[indices[scores_sort]]
                #没有检测出行人
                else:
                    image_boxes=np.zeros((0,4))
                    image_scores=0.0
                    image_labels=None
                # print(image_scores, image_labels, image_boxes)
                # draw_box(img, boxes, labels)
                # print (filename,type(filename))
                # print (filename.split('/'))
                write_file=filename.split('/')[-2:]
                write_file=write_file[0]+'/'+write_file[1]
                write_file=write_file.split('.')[0]
                for i in range(image_boxes.shape[0]):
                    # print (write_file)
                    # print (image_scores[i])
                    print (type(image_boxes[i, 0]))
                    # print (int(image_boxes[i, 1]))
                    # print (int(image_boxes[i, 2]))
                    # print (int(image_boxes[i, 3]))
                    f.write("%s %f %d %d %d %d\n"%(write_file,image_scores[i],int(image_boxes[i, 0]),int(image_boxes[i, 1]),int(image_boxes[i, 2]),int(image_boxes[i, 3])))




def generate_single_txt_result(img_path,retinanet,iou_threshold=0.3,score_threshold=0.5,max_detections=100):
    """
    输入是一个图片文件夹，输出是val.txt这个预测结果，然后再通过generate_citypersons_jsonResult.py文件转成可以evaluate的文件

    我们需要读取cache文件，然后将id与image_name进行对应

    :param img_path:
    :param retinanet:

    :param iou_threshold:
    :param score_threshold:
    :param max_detections:
    :return:
    """
    import pickle
    cache_path = 'data/cache/cityperson_py36/val'
    with open(cache_path, 'rb') as fid:
        val_data = pickle.load(fid)

    filenames=list_all_files(img_path)
    generate_txt_filename='predict_result/attention_val_epoch107.txt'#change
    print(len(val_data))
    with open(generate_txt_filename, 'w', encoding='utf-8') as Rep_f:

        #预测所有的图片
        # for filename in filenames:
        for f in range(len(val_data)):
            # print (type(filename))
            filename=val_data[f]['filepath']
            print(filename)
            frame_number = f + 1
            img=cv2.imread(filename)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            img = cv2.resize(img,(2048, 1024))

            if len(img.shape) == 2:
                img = skimage.color.gray2rgb(img)


            # height,width,_=img.shape
            # img=img.astype(np.float32) / 255.0
            # base_transform=transforms.Compose([
            # Resizer(), Normalizer()
            # ])

            data_img=base_transform(img)
            # print(data_img.shape)
            retinanet.eval()
            #开始进行测试
            with torch.no_grad():
                scores, labels, boxes = retinanet(data_img.cuda().float().unsqueeze(dim=0))

                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                    labels = labels.cpu().numpy()
                    boxes = boxes.cpu().numpy()

                    # add nms
                    index_cls = py_cpu_nms(boxes, scores, thresh=iou_threshold)
                    boxes = boxes[index_cls]
                    scores = scores[index_cls]
                    labels = labels[index_cls]

                    # print(boxes,scores,labels)
                    # print("############################")

                    # add Ot
                    # select indices which have a score above the threshold,Ot
                    indices = np.where(scores > score_threshold)[0]

                    # select those scores，根据indices过滤掉score值小于阈值的index
                    scores = scores[indices]

                    # find the order with which to sort the scores
                    scores_sort = np.argsort(-scores)[:max_detections]

                    # select detections,返回score个box
                    image_boxes = boxes[indices[scores_sort], :]
                    image_scores = scores[scores_sort]
                    image_labels = labels[indices[scores_sort]]
                #没有检测出行人
                else:
                    image_boxes=np.zeros((0,4))
                    image_scores=0.0
                    image_labels=None
                    frame_number=frame_number+1
                # print(image_scores, image_labels, image_boxes)
                # draw_box(img, boxes, labels)
                # print (filename,type(filename))
                # print (filename.split('/'))
                # write_file=filename.split('/')[-2:]
                # write_file=write_file[0]+'/'+write_file[1]
                # write_file=write_file.split('.')[0]
                for i in range(image_boxes.shape[0]):
                    # print (write_file)
                    # print (image_scores[i])
                    # print (type(image_boxes[i, 0]))
                    # print (int(image_boxes[i, 1]))
                    # print (int(image_boxes[i, 2]))
                    # print (int(image_boxes[i, 3]))
                    Rep_f.write("%d %d %d %d %d %f\n"%(frame_number,int(image_boxes[i, 0]),int(image_boxes[i, 1]),int(image_boxes[i, 2]-image_boxes[i, 0]),int(image_boxes[i, 3]-image_boxes[i, 1]),image_scores[i]))




def test_multi_img(img_path,retinanet,pic_output_path="",iou_threshold=0.3,score_threshold=0.5,max_detections=100):
    filenames = list_all_files(img_path)
    for filename in filenames:
        test_img(filename,retinanet,pic_output_path=pic_output_path)


img_file='/home/xuy/code/Repulsion_Loss/pic_val/frankfurt_000001_004327_leftImg8bit.png'
img_path='/home/xuy/dataset/citypersons/leftImg8bit/val'
# /home/xuy/dataset/citypersons/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png
model_file='/home/xuy/code/毕设相关代码/repulsion_loss_attention/ckpt/result_107.pt'#change
video_input_path='/home/xuy/code/Bi-box_Regression/demo.avi'
video_output_path='/home/xuy/code/Bi-box_Regression/result_0115.avi'
pic_output_path='/home/xuy/code/Repulsion_Loss/pic_val_result/'
####################
#load model
print('load model')
retinanet = torch.load(model_file)

# print(retinanet)
retinanet = torch.nn.DataParallel(retinanet, device_ids=[0])
retinanet.cuda()


"""
调用测试函数
"""
#测试单张图片
# test_img(img_file,retinanet,pic_output_path=pic_output_path)
#测试视频
# video_test(video_input_path,retinanet,output_path=video_output_path)
#测试多张图片
# test_multi_img(img_path,retinanet,pic_output_path=pic_output_path)
#生成txt文件，然后转化为json文件用来跑分测试
generate_single_txt_result(img_path,retinanet)
