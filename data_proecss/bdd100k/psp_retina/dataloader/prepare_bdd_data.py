import json
import os
import numpy as np
from .kmeans import kmeans, avg_iou

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
_width_ = 1280
_height_ = 720


def get_all_files(data_path, files, suffix):
    current_folder = data_path
    dirs = [dir for dir in os.listdir(data_path) if True == os.path.isdir(os.path.join(data_path, dir))]
    if len(dirs) == 0:
        file_list = os.listdir(current_folder)
        file_list_suffix = [os.path.join(current_folder, line) for line in file_list if
                    line[-len(suffix):] == suffix]
        files.extend(file_list_suffix)
    else:
        for dir in dirs:
            get_all_files(os.path.join(current_folder, dir), files, suffix)



def read_dataset(imagepath, bbox_trainfile, segpath):
    train_images = []
    seg_images = []
    get_all_files(imagepath, train_images, 'jpg')
    get_all_files(segpath, seg_images, 'png')
    bboxes = []
    with open(bbox_trainfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i in range(len(data)):
            label_txt = os.path.join(imagepath, data[i]['name'][:-3] + 'txt')
            fid_label = open(label_txt,'w')
            for j in range(len(data[i]['labels'])):
                label = data[i]['labels'][j]
                if not label_map.__contains__(label['category']):
                    continue
                cls = label_map[label['category']]
                x1 = label['box2d']['x1'] / _width_
                y1 = label['box2d']['y1'] / _height_
                x2 = label['box2d']['x2'] / _width_
                y2 = label['box2d']['y2'] / _height_
                bboxes.append([x2 - x1, y2 - y1])
                label_str = "%d %.6f %.6f %.6f %.6f\n"%(cls, x1, y1, x2, y2)
                fid_label.write(label_str)

            fid_label.close()
    return np.array(bboxes)

read_dataset('/home/test/xuy/challenageAI/bdd_dataset/images/100k/val',
             '/home/test/xuy/challenageAI/bdd_dataset/labels/bdd100k_labels_images_val.json',
             '/home/test/xuy/challenageAI/bdd_dataset/drivable_maps/labels/val')

CLUSTERS = 5
out = kmeans(data, k=CLUSTERS)

print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))
