# -*- coding:utf-8 -*- 
__author__ = 'xuy'

"""
最终版本的代码
生成coco的检测结果格式,可以替代dt_txt2json.m文件

使用city_res50_2step.hdf5生成的预测结果
 Average Miss Rate  (MR) @ Reasonable         [ IoU=0.50      | height=[50:10000000000] | visibility=[0.65:10000000000.00] ] = 12.19%
 Average Miss Rate  (MR) @ Reasonable_small   [ IoU=0.50      | height=[50:75] | visibility=[0.65:10000000000.00] ] = 43.91%
 Average Miss Rate  (MR) @ Reasonable_occ=heavy [ IoU=0.50      | height=[50:10000000000] | visibility=[0.20:0.65] ] = 44.09%
 Average Miss Rate  (MR) @ All                [ IoU=0.50      | height=[20:10000000000] | visibility=[0.20:10000000000.00] ] = 39.02%

"""
import copy
# import eval_demo


file_list = [

'/home/xuy/code/毕设相关代码/repulsion_loss_attention/predict_result/attention_val_epoch107.txt'#change

]

def generate_json(file_name):
    rf = open(file_name, "r")



    content = rf.readline()
    #用来保存所有的图片文件名，im_name
    img_list = []
    #一个字典文件，key是im_name，value是info_str，ｅｇ: 0.149 1214.6 359.7 1269.1 488.0
    img_dir  = {}
    #读取文件信息
    while content:
        num_id=int(float(content.strip('\n').split(' ')[0]))
        info_str=content.strip('\n').split(' ')[1:]
        if num_id in img_dir:
            img_dir[num_id].append(info_str)
        else:
            img_list.append(num_id)
            img_dir[num_id]=[]
            img_dir[num_id].append(info_str)
        content=rf.readline()

    # print img_dir
    print("paper dt_data done.")
    last_index=int(img_list[-1])
    # print last_index
    dt_coco = []
    for i in range(1,last_index+1):
        dt_info={}
        #如果判断出来当前id等于img_dir当中的key值了，那么就加入字典中
        for key,value in img_dir.items():
            if i==key:
                # print len(value)#获取每个id的框的个数
                for j in range(len(value)):

                    # print value[0][:4]
                    bbox,score=value[j][:4],value[j][4]
                    bbox=int(float(bbox[0])),int(float(bbox[1])),int(float(bbox[2])),int(float(bbox[3]))
                    score=float(score)
                    # print (i,bbox,score)
                    dt_info['image_id']=i
                    dt_info['category_id']=1
                    dt_info['bbox']=bbox
                    dt_info['score']=score
                    dt_coco.append(copy.deepcopy(dt_info))





    # print (len(dt_coco))
    import json

    jsObj = json.dumps(dt_coco)

    fileObject = open('/home/xuy/code/毕设相关代码/repulsion_loss_attention/evaluate/attention_epoch107.json', 'w')#change
    fileObject.write(jsObj)
    fileObject.close()



for file in file_list:
    generate_json(file)
    # eval_demo.run_eval_demo()