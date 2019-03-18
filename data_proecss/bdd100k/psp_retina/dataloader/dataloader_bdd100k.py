from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2

Dataset_folder = '/home/test/xuy/challenageAI/bdd_dataset/images/100k/train'

Dataset_val_folder='/home/test/xuy/challenageAI/bdd_dataset/images/100k/val'
NUM_CLASS = 11
label_map = {
    'bike':     1,
    'bus':      2,
    'car':      3,
    'motor':    4,
    'person':   5,
    'rider':    6,
    'traffic light':    7,
    'traffic sign':     8,
    'train':    9,
    'truck':    10
}
label_names = ['bg', 'bike', 'bus', 'car', 'motor', 'person', 'rider', 'light', 'sign', 'train', 'truck']

def get_all_files(data_path, files, suffix):
    current_folder = data_path
    dirs = [dir for dir in os.listdir(data_path) if True == os.path.isdir(os.path.join(data_path, dir))]
    if len(dirs) == 0:
        file_list = os.listdir(current_folder)
        file_list_suffix = [os.path.join(current_folder, line) for line in file_list if
                    line[-3:] == suffix]
        files.extend(file_list_suffix)
    else:
        for dir in dirs:
            get_all_files(os.path.join(current_folder, dir), files, suffix)


class DataAug(object):
    def __init__(self, random_flip=True,color_jitter=True):
        self.random_flip = random_flip
        # self.__width__ = 512
        # self.__height__ = 288
        self.__width__ = 720
        self.__height__ = 405
        self.color_jitter = None
        if color_jitter:
            self.color_jitter = transforms.ColorJitter(0.5, 0.15, 0.15, 0.1)

    def Augment(self, img, anno, map):
        if self.random_flip:
            flip = random.randint(0, 1)
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                map = map.transpose(Image.FLIP_LEFT_RIGHT)
                anno_centors = (anno[:,0] + anno[:,2])/2
                anno_centors = 1 - anno_centors
                anno_widths = anno[:,2] - anno[:,0]
                anno[:, 0] = anno_centors - anno_widths / 2
                anno[:, 2] = anno_centors + anno_widths / 2

        if self.color_jitter is not None:
            img = self.color_jitter(img)

        img = img.resize((int(self.__width__), int(self.__height__)))
        map = map.resize((int(self.__width__), int(self.__height__)))

        return img, anno, map


base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.456), (0.229,0.224,0.225))
    ])


def showpic(img, anno):
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    w,h = img.size
    for i in range(anno.shape[0]):
        cv2.rectangle(image,(int(anno[i,0]),int(anno[i,1])),
                      (int(anno[i,2]),int(anno[i,3])), (0,0,255))
    cv2.imshow('Image', image)
    cv2.waitKey(0)

# class BDD100kDataset_val(Dataset):
#     def __init__(self,Dataset_folder):
#         self.root_dir_val=Dataset_folder
#         self.base_transform = base_transform
#         self.imgs=[os.path.join(self.root_dir_val,img)for img in os.listdir(self.root_dir_val)]
#
#         self.img_nums=len(imgs)
#
#         normalize=self.base_transform
#
#         self.transforms=transforms.Compose([
#             transforms.Scale(224),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             normalize]
#         )
#     def __getitem__(self, idx):
#         img_path = Image.open(self.imgs[idx]).convert('RGB')
#
#         # print("!!!!!",self.image_list[idx])
#         mapfile = self.image_list[idx][:-4] + '_drivable_id.png'







class BDD100KDataset(Dataset):

    def __init__(self):
        self.root_dir = Dataset_folder
        self.base_transform = base_transform
        self.data_aug = DataAug(random_flip=True,color_jitter=True)
        self.multi_scale_train = 0.1
        self.anno_list = []
        get_all_files(self.root_dir, self.anno_list, 'txt')
        self.image_list = []
        self.annotations = []
        #change
        self.max_objs = 100
        for i in range(len(self.anno_list)):
            image_path = self.anno_list[i][:-3] + 'jpg'
            self.image_list.append(image_path)
            with open(self.anno_list[i], 'rb') as f:
                lines = f.readlines()
                annotation = []
                for line in lines:
                    splited = line.split()
                    cls = int(splited[0])
                    xmin = float(splited[1])
                    ymin = float(splited[2])
                    xmax = float(splited[3])
                    ymax = float(splited[4])
                    annotation.append([xmin, ymin, xmax, ymax, cls])
                self.annotations.append(annotation)

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, idx):

        img = Image.open(self.image_list[idx]).convert('RGB')

        #print("!!!!!",self.image_list[idx])
        mapfile = self.image_list[idx][:-4] + '_drivable_id.png'
        map = Image.open(mapfile)

        annot = torch.tensor(self.annotations[idx])

        img, annot, map = self.data_aug.Augment(img, annot, map)
        # showpic(img, annot)
        # img = self.base_transform(img)
        # _, h, w = img.shape
        # annot[:, [0, 2]] *= w
        # annot[:, [1, 3]] *= h

        # map = torch.from_numpy(np.array(map)).long()

        sample = {'img': img, 'anno_det': annot, 'anno_seg': map}

        return sample

    def num_classes(self):
        return 11


    def collate_fn(self, batch):
        n = len(batch)
        imgs = []
        bboxes = []
        maps = []

        if self.multi_scale_train:
            scale = 1 + random.uniform(-self.multi_scale_train, self.multi_scale_train)
            new_w = int(self.data_aug.__width__ * scale)
            new_h = int(self.data_aug.__height__ * scale)

        for i in range(n):
            img = batch[i]['img'].resize((new_w, new_h))

            map = batch[i]['anno_seg'].resize((new_w, new_h))
            map = torch.from_numpy(np.array(map)).long()

            batch[i]['anno_det'][:, [0, 2]] *= new_w
            batch[i]['anno_det'][:, [1, 3]] *= new_h

            imgs.append(self.base_transform(img))
            maps.append(map)

            #change
            # bboxes.append([batch[i]['anno_det']])
            bboxes_t = torch.zeros([1,self.max_objs,5])
            for j in range(len(batch[i]['anno_det'])):
                bboxes_t[0, j, 0] = batch[i]['anno_det'][j, 0]
                bboxes_t[0, j, 1] = batch[i]['anno_det'][j, 1]
                bboxes_t[0, j, 2] = batch[i]['anno_det'][j, 2]
                bboxes_t[0, j, 3] = batch[i]['anno_det'][j, 3]
                bboxes_t[0, j, 4] = batch[i]['anno_det'][j, 4]
            bboxes.append(bboxes_t)

            # showpic(img, batch[i]['anno_det'])

        return {'img':torch.stack(imgs),'anno_det':torch.stack(bboxes),'anno_seg':torch.stack(maps)}
