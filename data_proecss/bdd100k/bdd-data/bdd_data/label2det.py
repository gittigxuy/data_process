import argparse
import json

__author__ = 'Fisher Yu'
__copyright__ = 'Copyright (c) 2018, Fisher Yu'
__email__ = 'i@yf.io'
__license__ = 'BSD'
#提取了部分作为检测的label.json
#python3 -m bdd_data.label2det.py /home/xuy/桌面/code/python/challenageAI/bdd_dataset/labels/labels/bdd100k_labels_images_train.json  /home/xuy/桌面/code/python/challenageAI/bdd_dataset/labels/labels/detection_augtrain.json

def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('label_path', help='path to the label dir')
    parser.add_argument('det_path', help='path to output detection file')
    args = parser.parse_args()

    return args

'''
原来大的json文件作为输入，因为这里保存了图片的位置信息，进行了数据增强之后改变的仅仅是图片的名字，其他的没有变
根据json文件找图片，如果json当中有的图片信息，然而并没有找到该图片信息，那么就会报错
'''
def label2det(frames):
    boxes = list()
    for frame in frames:
        for label in frame['labels']:
            if 'box2d' not in label:
                continue
            xy = label['box2d']
            if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                continue
            box_original = {'name': frame['name'],#filename
                   'timestamp': frame['timestamp'],
                   'category': label['category'],
                   'bbox': [xy['x1'], xy['y1'], xy['x2'], xy['y2']],
                   'score': 1}
            box_randomColor0={'name':frame['name'].split('.')[0]+'randomColor0'+'.jpg',
                   'timestamp': frame['timestamp'],
                   'category': label['category'],
                   'bbox': [xy['x1'], xy['y1'], xy['x2'], xy['y2']],
                   'score': 1}

            box_randomColor1={'name':frame['name'].split('.')[0]+'randomColor1'+'.jpg',
                   'timestamp': frame['timestamp'],
                   'category': label['category'],
                   'bbox': [xy['x1'], xy['y1'], xy['x2'], xy['y2']],
                   'score': 1}
            box_randomColor2 = {'name': frame['name'].split('.')[0]+'randomColor2'+'.jpg',
                                'timestamp': frame['timestamp'],
                                'category': label['category'],
                                'bbox': [xy['x1'], xy['y1'], xy['x2'], xy['y2']],
                                'score': 1}
            box_randomColor3 = {'name': frame['name'].split('.')[0]+'randomColor3'+'.jpg',
                                'timestamp': frame['timestamp'],
                                'category': label['category'],
                                'bbox': [xy['x1'], xy['y1'], xy['x2'], xy['y2']],
                                'score': 1}
            boxes.append(box_original)
            boxes.append(box_randomColor0)
            boxes.append(box_randomColor1)
            boxes.append(box_randomColor2)
            boxes.append(box_randomColor3)



    return boxes


def convert_labels(label_path, det_path):
    frames = json.load(open(label_path, 'r'))
    det = label2det(frames)
    json.dump(det, open(det_path, 'w'), indent=4, separators=(',', ': '))


def main():
    args = parse_args()
    convert_labels(args.label_path, args.det_path)


if __name__ == '__main__':
    main()

"""

"""
