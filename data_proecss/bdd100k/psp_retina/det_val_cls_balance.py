import numpy as np
import torch
from dataloader.dataloader_bdd100k import base_transform, NUM_CLASS, label_names, get_all_files
from models.resnet_psp_retina.model import resnet50
from PIL import Image
import cv2
from train_retina_psp_res50 import anchor_config, Pyramid_Feature_Size, Cls_Feature_Size, Reg_Feature_Size,Psp_In_Feature_Size,Psp_Out_Feature_Size
from utils.utils import py_cpu_nms
import os
import json
import time
# from utils.nms_wrapper import nms
# input_w = 512
# input_h = 288
from PIL import Image,ImageDraw
input_w = 720
input_h = 405

gt_labels = '/home/xuy/桌面/code/python/challenageAI/bdd_dataset/labels/labels/detection_val.json'
val_folder = '/home/xuy/桌面/code/python/challenageAI/bdd_dataset/images/images/100k/val/'
model_path = '/home/xuy/code/psp_retina_2033_785/weights/retinanet_24.pth'#change





val_seg_result_path='/home/xuy/code/psp_retina_2033_785'
os.makedirs(val_seg_result_path+'/seg_epoch24/',exist_ok=True)#change





def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc




def cls_nms(transformed_anchors, scores, clss, num_classes):
    bboxes = []
    probs = []
    cls_return = []
    for i in range(num_classes):
        cls = i + 1
        index = np.where(clss==cls)[0]
        if len(index) == 0:
            continue
        bboxes_cls = transformed_anchors[index]
        scores_cls = scores[index]

        c_dets = np.hstack((bboxes_cls, scores_cls[:, np.newaxis])).astype(np.float32, copy=False)

        index_cls = py_cpu_nms(bboxes_cls, scores_cls)
        # index_cls = nms(c_dets, 0.45,force_cpu=True)
        # index_cls = soft_nms(bboxes_cls, scores_cls)
        if len(bboxes)==0:
            bboxes = bboxes_cls[index_cls]
            probs = scores_cls[index_cls]
            cls_return = np.ones(len(index_cls))*cls
        else:
            bboxes = np.vstack((bboxes, bboxes_cls[index_cls]))
            probs = np.hstack((probs, scores_cls[index_cls]))
            cls_return = np.hstack((cls_return,np.ones(len(index_cls))*cls))
    return np.array(probs), np.array(cls_return), np.array(bboxes)


def preprocess_pil(image,w,h):
    if image.mode != 'RGB':
        image.convert('RGB')

    scale_w = image.size[0] / w
    scale_h = image.size[1] / h
    image = image.resize((w, h))
    return base_transform(image), image, [scale_w, scale_h]


def prepare_test_input(image_path ,w,h):
    image = Image.open(image_path)
    data, image, scales = preprocess_pil(image, w,h)
    return data, image, scales


def draw_box(img, boxes,labels, waitkey_value=0):
    cvimg = np.array(img)
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
    for i in range(boxes.shape[0]):
        cv2.rectangle(cvimg,(int(boxes[i,0]),int(boxes[i,1])),(int(boxes[i,2]),int(boxes[i,3])),(0,255,0) )
        cv2.putText(cvimg,label_names[labels[i]],(int(boxes[i,0]),int(boxes[i,1])),cv2.FONT_HERSHEY_COMPLEX,.5,(0,0,255),1)
    cv2.imshow('Detection', cvimg)
    key = cv2.waitKey(waitkey_value)
    return key


def draw_maps(maps, waitkey_value=0):
    maps_show = np.zeros(maps.shape)
    for h in range(maps.shape[0]):
        for w in range(maps.shape[1]):
            cls = np.argmax(maps[h,w,:])
            maps_show[h,w,cls] = maps[h,w,cls]
    maps_show *= 255
    cv2.imshow('Segmentation', maps_show)
    key = cv2.waitKey(waitkey_value)


def image_test(image_path, model, showRes=False, thresh=0.5, save_json=False,save_mask=True):
    model.eval()
    boxes_list=[]

    with torch.no_grad():
        start_time=time.time()
        data, image, scales = prepare_test_input(image_path,input_w,input_h)
        scores, labels, boxes, maps = model(data.cuda().unsqueeze(dim=0),thresh)
        #add this line to use nms to each classify
        scores, labels, boxes = cls_nms(boxes,scores,labels,NUM_CLASS)
        if showRes:
            draw_box(image, boxes, labels)
            draw_maps(maps)
        if save_mask:
            #draw mask
            mask_pred=np.argmax(maps,axis=-1).astype(np.uint8)
            mask_img=Image.fromarray(mask_pred).resize((1280,720),Image.NEAREST)
            mask_img.save(val_seg_result_path+'/seg_epoch24/'+os.path.basename(image_path).replace('.jpg','_drivable_id.png'))#change

            # print('!!!!mask per image:', time.time() - start_time)

        if save_json:
            # save_txt_path=val_result_path+os.path.basename(image_path)[:-3]+'txt'
            # save_txt_path=os.path.join(val_result_path,os.path.basename(image_path)[:-3]+'txt')
            # with open(save_txt_path,'w',encoding='utf-8') as f:
            #     for i in range(len(labels)):
            #         f.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f\n"%(label_names[labels[i]], scores[i], boxes[i,0]*scales[0], boxes[i,1]*scales[1],
            #                                                  boxes[i,2]*scales[0], boxes[i,3]*scales[1]))

            for i in range(len(labels)):

                filename=os.path.basename(image_path)
                # print(filename)
                class_label,score,xmin,ymin,xmax,ymax=label_names[int(labels[i])],scores[i],boxes[i,0]*scales[0], boxes[i,1]*scales[1],boxes[i,2]*scales[0],boxes[i,3]*scales[1]
                # print("!!!",class_label)
                if class_label == 'sign':
                    class_label = 'traffic sign'
                if class_label == 'light':
                    class_label = 'traffic light'

                box = {
                    'name': filename,
                    'timestamp': 1000,
                    'category': class_label,
                    'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)],
                    'score': float(score)

                }
                # print(type(boxes),type(box))
                boxes_list.append(box)
            # print('!!!!detection per image:', time.time() - start_time)
            return boxes_list


#
# def image_test_ms(image_path, model, showRes=False, thresh=0.5,save_json=False,save_mask=True):
#     boxes_list=[]
#     multi_scales = [720, 640, 512]
#     scale = input_w / input_h
#     with torch.no_grad():
#         boxes_ms = []
#         scores_ms = []
#         labels_ms = []
#         maps = []
#         for i in range(len(multi_scales)):
#             data, image, scales = prepare_test_input(image_path, multi_scales[i], int(multi_scales[i] / scale))
#             scores, labels, boxes, maps = model(data.cuda().unsqueeze(dim=0),thresh)
#             boxes[:, 0] = boxes[:, 0] * scales[0]
#             boxes[:, 1] = boxes[:, 1] * scales[1]
#             boxes[:, 2] = boxes[:, 2] * scales[0]
#             boxes[:, 3] = boxes[:, 3] * scales[1]
#             if len(boxes_ms) == 0:
#                 boxes_ms = boxes
#                 scores_ms = scores
#                 labels_ms = labels
#             else:
#                 boxes_ms = np.vstack((boxes_ms, boxes))
#                 scores_ms = np.hstack((scores_ms, scores))
#                 labels_ms = np.hstack((labels_ms, labels))
#
#         scores, labels, boxes = cls_nms(boxes_ms, scores_ms, labels_ms, NUM_CLASS)
#         image = Image.open(image_path)
#         if showRes:
#             draw_box(image, boxes, labels)
#             draw_maps(maps)
#         if save_mask:
#             #draw mask
#             mask_pred=np.argmax(maps,axis=-1).astype(np.uint8)
#             mask_img=Image.fromarray(mask_pred).resize((1280,720),Image.NEAREST)
#             mask_img.save(val_seg_result_path+'/seg_epoch18/'+os.path.basename(image_path).replace('.jpg','_drivable_id.png'))#change
#
#             # print('!!!!mask per image:', time.time() - start_time)
#
#         if save_json:
#             # save_txt_path=val_result_path+os.path.basename(image_path)[:-3]+'txt'
#             # save_txt_path=os.path.join(val_result_path,os.path.basename(image_path)[:-3]+'txt')
#             # with open(save_txt_path,'w',encoding='utf-8') as f:
#             #     for i in range(len(labels)):
#             #         f.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f\n"%(label_names[labels[i]], scores[i], boxes[i,0]*scales[0], boxes[i,1]*scales[1],
#             #                                                  boxes[i,2]*scales[0], boxes[i,3]*scales[1]))
#
#             for i in range(len(labels)):
#
#                 filename=os.path.basename(image_path)
#                 # print(filename)
#                 class_label,score,xmin,ymin,xmax,ymax=label_names[int(labels[i])],scores[i],boxes[i,0]*scales[0], boxes[i,1]*scales[1],boxes[i,2]*scales[0],boxes[i,3]*scales[1]
#                 # print("!!!",class_label)
#                 if class_label == 'sign':
#                     class_label = 'traffic sign'
#                 if class_label == 'light':
#                     class_label = 'traffic light'
#
#                 box = {
#                     'name': filename,
#                     'timestamp': 1000,
#                     'category': class_label,
#                     'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)],
#                     'score': float(score)
#
#                 }
#                 # print(type(boxes),type(box))
#                 boxes_list.append(box)
#             # print('!!!!detection per image:', time.time() - start_time)
#             return boxes_list


def init_model(class_num, trained_model_path):
    net = resnet50(num_classes_det=NUM_CLASS, num_classes_seg=3,
                   anchor_config=anchor_config, Pyramid_Feature_Size=Pyramid_Feature_Size,
                   Cls_Feature_Size=Cls_Feature_Size, Reg_Feature_Size=Reg_Feature_Size,
                   Psp_In_Feature_Size=Psp_In_Feature_Size, Psp_Out_Feature_Size=Psp_Out_Feature_Size,
                   pretrained=False).cuda()



    #not fix reference:https://www.ptorch.com/news/74.html
    state_dict = torch.load(trained_model_path)
    net.load_state_dict(state_dict)
    # # create new OrderedDict that does not contain `module.`
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k,v in state_dict.items():
    #     name=k.replace('module.','')
    #     new_state_dict[name] = v



    # net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(trained_model_path)['state_dict'].items()})


    # net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(trained_model_path).items()})


    return net


def generate_result_file():
    #load the model and evaluate detection result

    generate_json_path='/home/xuy/code/psp_retina_2033_785/json_result/retina_epoch24.json'#change


    val_list = []
    get_all_files(val_folder, val_list, 'jpg')


    net = init_model(NUM_CLASS, model_path)
    net.eval()

    json_boxes=[]

    for item in val_list:
        #print("!!!!",item)
        boxes=image_test(item, net, showRes=False, thresh=0.1, save_json=True,save_mask=True)
        # print(boxes)
        json_boxes.extend(boxes)
    json.dump(json_boxes,open(generate_json_path,'w'),indent=4,
              separators=(',', ': '))

def evaluate_mask():
    # val_folder = '/home/lab-yao.yuehan/Workspace/competitions/bdd100k/data/bdd100k/images/100k/val/'
    val_list = []

    get_all_files(val_folder, val_list, 'jpg')
    m1 = False
    if m1:
        accuracy = 0
        mIU = 0
    else:
        n_class = 3
        hist = np.zeros((n_class, n_class))
    for idx,image_path in enumerate(val_list):
        pred_fname = val_seg_result_path+'/seg_epoch24/'+os.path.basename(image_path).replace('.jpg','_drivable_id.png')#change
        mask_fname = image_path.replace('.jpg','_drivable_id.png').replace('images/100k/','drivable_maps/labels/')
        labels_true = cv2.imread(mask_fname,cv2.IMREAD_GRAYSCALE)
        predictions = cv2.imread(pred_fname,cv2.IMREAD_GRAYSCALE)

        if m1:
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(labels_true,predictions,3)
            accuracy += acc_cls
            mIU +=mean_iu
        else:
            hist += _fast_hist(labels_true.flatten(), predictions.flatten(), n_class)

        if idx >1000:
            break
    if m1:
        print('acc:',accuracy/(idx+1))
        print('mIou:',mIU/(idx+1))
    else:
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        mean_acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        print('acc_cls:{} iu_cls:{} iu:{}'.format(acc_cls,iu,sum(iu[1:])/2))

if __name__ == '__main__':
    generate_result_file()
    # evaluate_mask()
#
#
#
#
# import json
# import os
# def iterbrowse(path):
#     for home, dirs, files in os.walk(path):
#         for filename in files:
#             yield os.path.join(home, filename)
#
#
# def get_all_files(data_path, files, suffix):
#     current_folder = data_path
#     dirs = [dir for dir in os.listdir(data_path) if True == os.path.isdir(os.path.join(data_path, dir))]
#     if len(dirs) == 0:
#         file_list = os.listdir(current_folder)
#         file_list_suffix = [os.path.join(current_folder, line) for line in file_list if
#                     line[-3:] == suffix]
#         files.extend(file_list_suffix)
#     else:
#         for dir in dirs:
#             get_all_files(os.path.join(current_folder, dir), files, suffix)
# def json2txt(txt_filepath):
#     boxes = []
#     txt_list =[]
#     get_all_files(txt_filepath, txt_list, 'txt')
#     # for txt_path in iterbrowse(txt_filepath):
#     for txt_path in txt_list:
#         txt_basename=os.path.basename(txt_path)
#         portion = os.path.splitext(txt_basename)
#         if portion[1] == '.txt':
#             pic_filename = portion[0] + '.jpg'
#         txt_file=open(txt_path,encoding='utf-8')
#         lines=txt_file.readlines()
#
#         for line in lines:
#             line=line.split(',')
#             # class_label, prob, xmin, ymin, xmax, ymax = line[-6], line[-5], line[-4], line[-3], line[-2], line[-1]
#             class_label, prob, xmin, ymin, xmax, ymax = line[0], line[1], line[2], line[3], line[4], line[5]
#             if class_label=='sign':
#                 class_label='traffic sign'
#             if class_label=='light':
#                 class_label = 'traffic light'
#
#             box = {
#                 'name': pic_filename,
#                 'timestamp': 1000,
#                 'category': class_label,
#                 'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)],
#                 'score': float(prob)
#
#             }
#             boxes.append(box)
#
#     return boxes





    # det = json2txt(val_result_path)
    # json.dump(det, open('/home/test/xuy/pytorch_retina_1105/json_result/retina_1110_2.json', 'w'), indent=4,
    #           separators=(',', ': '))
