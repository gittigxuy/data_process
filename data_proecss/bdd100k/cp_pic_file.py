# -*- coding:utf-8 -*- 
__author__ = 'xuy'

import shutil

def objFilename():
    #存储的是文件名的前缀
    local_file_name_list='test.txt'
    obj_name_list=[]
    for i in open(local_file_name_list,'r'):
        obj_name_list.append(i.replace('\n',''))
    return obj_name_list

def copy_img():
    #存储原图片文件夹路径
    local_img_name='/home/xuy/桌面/code/python/challenageAI/bdd_dataset/images/images/100k/test'
    #指定目标路径
    dst_path='/home/xuy/桌面/code/python/challenageAI/pic_test'
    for i in objFilename():
        new_obj_name=i+'.jpg'
        shutil.copy(local_img_name+'/'+new_obj_name,dst_path+'/'+new_obj_name)

copy_img()