# -*- coding:utf-8 -*-
__author__ = 'xuy'
import os
import pickle

import numpy as np
from scipy import io as scio

from my_bbox import box_op

"""
run this file to generate cache,then run test.py

or
使用这个代码，可以将原来的train.m或者test.m转化为Pickle类型
you could prepare the data according this code,to prepare the data structure


"""
#####

#######

root_dir = 'data/cityperson'
all_img_path = os.path.join(root_dir, 'images')
all_anno_path = os.path.join(root_dir, 'annotations')
types = ['train']
rows, cols = 1024, 2048
for type in types:
	anno_path = os.path.join(all_anno_path, 'anno_'+type+'.mat')
	res_path = os.path.join('data/cache/cityperson', type)
	image_data = []
	annos = scio.loadmat(anno_path)
	index = 'anno_'+type+'_aligned'
	valid_count = 0
	iggt_count = 0
	rea_count = 0
	box_count = 0
	for l in range(len(annos[index][0])):
		anno = annos[index][0][l]
		# cityname = anno[0][0][0][0].encode()
		# imgname = anno[0][0][1][0].encode()

		cityname = anno[0][0][0][0]
		imgname = anno[0][0][1][0]
		gts = anno[0][0][2]
		# print(all_img_path)
		# print(type)
		# print(cityname)
		# print(imgname)
		img_path = os.path.join(all_img_path, type + '/'+ cityname+'/'+imgname)
		boxes = []
		ig_boxes = []
		vis_boxes = []
		# if len(gts)==0:
		# 	continue
		for i in range(len(gts)):
			label, x1, y1, w, h = gts[i, :5]
			x1, y1 = max(int(x1), 0), max(int(y1), 0)
			w, h = min(int(w), cols - x1 -1), min(int(h), rows - y1 -1)
			xv1, yv1, wv, hv = gts[i, 6:]#vis_part
			xv1, yv1 = max(int(xv1), 0), max(int(yv1), 0)
			wv, hv = min(int(wv), cols - xv1 - 1), min(int(hv), rows - yv1 - 1)
			# here we only choose height>50 optionally
			if label == 1 and h>=50:
				rea_count += 1
				box = np.array([int(x1), int(y1), int(x1)+int(w), int(y1)+int(h)])#x1,y1,x2,y2
				boxes.append(box)
				vis_box = np.array([int(xv1), int(yv1), int(xv1)+int(wv), int(yv1)+int(hv)])
				vis_boxes.append(vis_box)
			else:#ignore part
				ig_box = np.array([int(x1), int(y1), int(x1)+int(w), int(y1)+int(h)])
				ig_boxes.append(ig_box)
		boxes = np.array(boxes)
		vis_boxes = np.array(vis_boxes)
		ig_boxes = np.array(ig_boxes)
		# if len(boxes)==0:
		# 	continue
		valid_count += 1
		# img = cv2.imread(img_path)
		# for i in range(len(boxes)):
		# 	(x1, y1, x2, y2) = boxes[i, :]
		# 	if y2-y1>50:
		# 		cv2.rectangle(img, (x1, y1), (x2, y2),(255,0,0), 2)
		# 		y3 = int(y1+(y2-y1)/3.0)
		# 		cv2.rectangle(img, (x1, y1), (x2, y3), (0, 255, 0), 2)
		# 		# cv2.putText(img, str(boxes[i, 0]), (x1 - 2, y1 - 2), cv2.FONT_HERSHEY_DUPLEX, 1,colors[label], 1)
		# 		plt.imshow(img, interpolation='bicubic')
		annotation = {}
		annotation['filepath'] = img_path
		if len(ig_boxes) > 0 and len(boxes) > 0:
			boxig_overlap = box_op(np.ascontiguousarray(boxes, dtype=np.float),
								   np.ascontiguousarray(ig_boxes, dtype=np.float))
			ignore_sum = np.sum(boxig_overlap, axis=1)
			oriboxes = np.copy(boxes)
			boxes = oriboxes[ignore_sum < 0.5, :]
			vis_boxes = vis_boxes[ignore_sum < 0.5, :]

			if ignore_sum.max()>=0.5:
				iggt_count += len(ignore_sum)-len(boxes)
				ig_boxes = np.concatenate([ig_boxes, oriboxes[ignore_sum >= 0.5, :]], axis=-0)
		box_count += len(boxes)
		annotation['bboxes'] = boxes
		annotation['vis_bboxes'] = vis_boxes
		annotation['ignoreareas'] = ig_boxes
		image_data.append(annotation)
	with open(res_path, 'wb') as fid:
		pickle.dump(image_data, fid, pickle.HIGHEST_PROTOCOL)
	print ('{} has {} images and {} valid images, {} valid gt and {} ignored gt'.format(type, len(annos[index][0]), valid_count, box_count, iggt_count))
