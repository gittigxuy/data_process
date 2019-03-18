import numpy as np
import torch
from dataloader.dataloader_bdd100k import base_transform, NUM_CLASS, label_names
from models.resnet_psp_retina.model import resnet50
from PIL import Image
import cv2
from utils.utils import py_cpu_nms
# from train_retina_psp_res50 import anchor_config, Pyramid_Feature_Size, Cls_Feature_Size, Reg_Feature_Size,Psp_In_Feature_Size,Psp_Out_Feature_Size
input_w = 720
input_h = 405


anchor_config = {
	'ratios': [0.5, 1, 2],
	'scales': [0.5, 0.75, 1, 1.25],
	'sizes': [16 ,32, 64, 128, 256]
}

Pyramid_Feature_Size = 256
Cls_Feature_Size = 256
Reg_Feature_Size = 256
Psp_In_Feature_Size = 512
Psp_Out_Feature_Size = 256
NUM_CLASS = 11

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
        #默认Nt=0.3
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
    # return probs, cls_return, bboxes

def preprocess_pil(image):
    if image.mode != 'RGB':
        image.convert('RGB')
    im_w, im_h = image.size
    scale = im_w * 1. / im_h
    new_h = 288
    new_w = new_h * scale
    if (new_w > 512):
        new_w = 512
        new_h = new_w / scale
    image = image.resize((int(new_w), int(new_h)))
    return base_transform(image), image


def prepare_test_input(image_path):
    image = Image.open(image_path)
    data, image = preprocess_pil(image)
    return data, image


def draw_box(img, boxes,labels, waitkey_value=0):
    cvimg = np.array(img)
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
    #i 表示一张图当中每一个框
    for i in range(boxes.shape[0]):
        cv2.rectangle(cvimg,(int(boxes[i,0]),int(boxes[i,1])),(int(boxes[i,2]),int(boxes[i,3])),(0,255,0),1 )
        cv2.putText(cvimg,
                    label_names[int(labels[i])],
                    (int(boxes[i,0]),int(boxes[i,1])),
                    cv2.FONT_HERSHEY_COMPLEX,.5,(0,0,255),1)
    #detection
    # cv2.resizeWindow(cvimg,500,500)
    cv2.imshow('Detection', cvimg)
    cv2.imwrite(filename='/home/xuy/code/psp_retina_2033_785/result.jpg',img=cvimg)

    key = cv2.waitKey(waitkey_value)
    return key

def draw_maps(maps, waitkey_value=0):
    maps_show = np.zeros(maps.shape)
    for h in range(maps.shape[0]):
        for w in range(maps.shape[1]):
            cls = np.argmax(maps[h,w,:])
            maps_show[h,w,cls] = maps[h,w,cls]
    maps_show *= 255
    #mask
    cv2.imshow('Segmentation', maps_show)
    key = cv2.waitKey(waitkey_value)
    return key

#add this function
def draw_box_maps(img, boxes,labels,maps,waitkey_value=0):
    color = ((0, 255, 0), (255, 0, 0))


    cvimg = np.array(img)
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)

    # maps_show = np.zeros(maps.shape)#fix
    for i in range(boxes.shape[0]):
        cv2.rectangle(cvimg, (int(boxes[i, 0]), int(boxes[i, 1])), (int(boxes[i, 2]), int(boxes[i, 3])), (0, 255, 0), 1)
        cv2.putText(cvimg, label_names[int(labels[i])], (int(boxes[i, 0]), int(boxes[i, 1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 0, 255), 1)

    for h in range(maps.shape[0]):
        for w in range(maps.shape[1]):
            cls = np.argmax(maps[h,w,:])
            # print("!!!",maps[h,w,cls])

            # cvimg[h,w,cls] = maps[h,w,cls]#fix
            # print("!!!",cvimg[h,w,cls])
            # cvimg[h,w,:] = [255,0,0]#fix



            if cls==1:
                cvimg[h, w, :] = color[0] # fix
            elif cls==2:
                cvimg[h, w, :]=color[1]
            else:
                pass


    # maps_show *= 255
    cv2.imshow('DetectSegmentation', cvimg)
    cvimg_result=cvimg
    key = cv2.waitKey(waitkey_value)
    return cvimg_result,key



def init_model(class_num, trained_model_path):
    net = resnet50(num_classes_det=NUM_CLASS, num_classes_seg=3,
                   anchor_config=anchor_config, Pyramid_Feature_Size=Pyramid_Feature_Size,
                   Cls_Feature_Size=Cls_Feature_Size,  Reg_Feature_Size=Reg_Feature_Size,
                   Psp_In_Feature_Size=Psp_In_Feature_Size,Psp_Out_Feature_Size=Psp_Out_Feature_Size,
                   pretrained=True).cuda()
    state_dict = torch.load(trained_model_path)
    net.load_state_dict(state_dict)
    return net


def image_test(image_path, model, showRes=False, thresh=0.5):
    model.eval()

    with torch.no_grad():
        data, image = prepare_test_input(image_path)
        scores, labels, boxes, maps = model(data.cuda().unsqueeze(dim=0),thresh)
        scores, labels, boxes = cls_nms(boxes, scores, labels, NUM_CLASS)#add NMS
        if showRes:
            draw_box(image, boxes, labels)
            draw_maps(maps)
def video_test(video_path, model, thresh=0.5,output_path=""):
    model.eval()

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    # im = preprocess_pil(frame)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    data, image = preprocess_pil(img)
    size = image.size
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")
    # video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
    # video_fps = cap.get(cv2.CAP_PROP_FPS)
    # video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, codec,25.0, size)
    while True:
        ret, frame = cap.read()
        img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        data, image = preprocess_pil(img)

        with torch.no_grad():
            scores, labels, boxes,maps = model(data.cuda().unsqueeze(dim=0), thresh)
            scores, labels, boxes = cls_nms(boxes, scores, labels, NUM_CLASS)
        # key = draw_box(image, boxes, labels,waitkey_value=1)
        # key=draw_maps(maps,waitkey_value=1)
        cvimg_result,key=draw_box_maps(image, boxes,labels,maps,waitkey_value=1)#fix
        cvimg_result=np.asarray(cvimg_result)
        if isOutput:
            out.write(cvimg_result)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

#demo image test
def demo_image_test():
    image_path = '/home/xuy/桌面/code/python/challenageAI/bdd_dataset/images/images/100k/test/db76adbc-855a5c91.jpg'
    model_path = '/home/xuy/code/psp_retina_2033_785/weights/retinanet_24.pth'
    net = init_model(NUM_CLASS, model_path)
    #这里的thresh表示Ot，就是score的阈值
    image_test(image_path, net, showRes=True, thresh=0.3)

#demo video test
def demo_video_test():
    # video_path = 'D:/dataset/detection_data/MP4/ch07_20170204085422.mp4'
    # video_path = '/home/xuy/code/psp_retina_2033_785/demo.mp4'
    video_path = '/home/xuy/code/psp_retina_2033_785/7f32eda9d2ccb45cb12406dbc1070a1f.mp4'
    model_path = '/home/xuy/code/psp_retina_2033_785/weights/retinanet_24.pth'
    net = init_model(NUM_CLASS, model_path)
    video_test(video_path, net, thresh=0.3,output_path='/home/xuy/code/psp_retina_2033_785/result_1223.mp4')

if __name__ == '__main__':
    # demo_image_test()
    demo_video_test()