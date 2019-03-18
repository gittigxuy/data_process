#!/usr/bin/python
# -*- coding:utf8 -*-


# from keras.utils import np_utils
# from keras.utils import plot_model
from .base_model import Base_model
from keras.optimizers import Adam
from keras.models import Model
from keras_alfnet.parallel_model import ParallelModel
from keras.utils import generic_utils
from keras_alfnet import losses as losses
from keras_alfnet import bbox_process
from . import model_alf
import numpy as np
import time, os, cv2
import cv2
from keras_alfnet import config
C = config.Config()
import matplotlib.pyplot as plt
import copy



def findMaxPath(links,dets):
    maxpaths=[] #保存从每个结点到最后的最大路径与分数
    roots=[] #保存所有的可作为独立路径进行最大路径比较的路径
    maxpaths.append([ (box[4],[ind]) for ind,box in enumerate(dets[-1])])
    for link_ind,link in enumerate(links[::-1]): #每一帧与后一帧的link，为一个list
        curmaxpaths=[]
        linkflags=np.zeros(len(maxpaths[0]),int)
        det_ind=len(links)-link_ind-1
        for ind,linkboxes in enumerate(link): #每一帧中每个box的link，为一个list
            if linkboxes == []:
                curmaxpaths.append((dets[det_ind][ind][4],[ind]))
                continue
            linkflags[linkboxes]=1
            prev_ind=np.argmax([maxpaths[0][linkbox][0] for linkbox in linkboxes])
            prev_score=maxpaths[0][linkboxes[prev_ind]][0]
            prev_path=copy.copy(maxpaths[0][linkboxes[prev_ind]][1])
            prev_path.insert(0,ind)
            curmaxpaths.append((dets[det_ind][ind][4]+prev_score,prev_path))
        root=[maxpaths[0][ind] for ind,flag in enumerate(linkflags) if flag == 0]
        roots.insert(0,root)
        maxpaths.insert(0,curmaxpaths)
    roots.insert(0,maxpaths[0])
    maxscore=0
    maxpath=[]
    for index,paths in enumerate(roots):
        if paths==[]:
            continue
        maxindex=np.argmax([path[0] for path in paths])
        if paths[maxindex][0]>maxscore:
            maxscore=paths[maxindex][0]
            maxpath=paths[maxindex][1]
            rootindex=index
    return rootindex,maxpath,maxscore

def rescore(dets, rootindex, maxpath, maxsum):
    newscore=maxsum/len(maxpath)
    for i,box_ind in enumerate(maxpath):
        dets[rootindex+i][box_ind][4]=newscore

def deleteLink(dets,links, rootindex, maxpath,thesh):
    for i,box_ind in enumerate(maxpath):
        areas=[(box[2]-box[0]+1)*(box[3]-box[1]+1) for box in dets[rootindex+i]]
        area1=areas[box_ind]
        box1=dets[rootindex+i][box_ind]
        x1=np.maximum(box1[0],dets[rootindex+i][:,0])
        y1=np.maximum(box1[1],dets[rootindex+i][:,1])
        x2=np.minimum(box1[2],dets[rootindex+i][:,2])
        y2=np.minimum(box1[3],dets[rootindex+i][:,3])
        w =np.maximum(0.0, x2 - x1 + 1)
        h =np.maximum(0.0, y2 - y1 + 1)
        inter = w * h
        ovrs = inter / (area1 + areas - inter)
        deletes=[ovr_ind for ovr_ind,ovr in enumerate(ovrs) if ovr >= C.overlap_thresh] #保存待删除的box的index
        if rootindex+i<len(links): #除了最后一帧，置box_ind的box的link为空
            for delete_ind in deletes:
                links[rootindex+i][delete_ind]=[]
        if i > 0 or rootindex>0:
            for priorbox in links[rootindex+i-1]: #将前一帧指向box_ind的link删除
                for delete_ind in deletes:
                    if delete_ind in priorbox:
                        priorbox.remove(delete_ind)


def createLinks(dets_all):
	links_all=[]
	frame_num=len(dets_all)
	link_start=time.time()
	for frame_index in range(frame_num-1):
		dets1=dets_all[frame_index]
		dets2=dets_all[frame_index+1]

		box1_num = len(dets1)
		box2_num = len(dets2)
		# 先计算每个box的area
		if frame_index == 0:
			areas1 = np.empty(box1_num)
			for box1_ind, box1 in enumerate(dets1):
				areas1[box1_ind] = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
		else:  # 当前帧的area1就是前一帧的area2，避免重复计算
			areas1 = areas2
		areas2 = np.empty(box2_num)
		for box2_ind, box2 in enumerate(dets2):
			areas2[box2_ind] = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
		# 计算相邻两帧同一类的link
		links_frame = []  # 保存相邻两帧的links
		#第一个帧的框
		for box1_ind, box1 in enumerate(dets1):
			area1 = areas1[box1_ind]
			"""
			>>> b = [i[0] for i in a]     # 从a中的每一行取第一个元素。
			>>>print(b)
			[1, 4]
			"""

			#遍历dets1，然后将dets2的第一个维度设置为全部的index


			# x1 = np.maximum(box1[0], dets2[box1_ind][0])
			# y1 = np.maximum(box1[1], dets2[box1_ind][1])
			# x2 = np.minimum(box1[2], dets2[box1_ind][2])
			# y2 = np.minimum(box1[3], dets2[box1_ind][3])

			x1 = np.maximum(box1[0], [i[0] for i in dets2])
			y1 = np.maximum(box1[1], [i[1] for i in dets2])
			x2 = np.minimum(box1[2], [i[2] for i in dets2])
			y2 = np.minimum(box1[3], [i[3] for i in dets2])

			w = np.maximum(0.0, x2 - x1 + 1)
			h = np.maximum(0.0, y2 - y1 + 1)
			inter = w * h
			ovrs = inter / (area1 + areas2 - inter)
			links_box = [ovr_ind for ovr_ind, ovr in enumerate(ovrs) if ovr >= C.overlap_thresh]  # 保存第一帧的一个box对第二帧全部box的link
			links_frame.append(links_box)
		links_all.append(links_frame)
	link_end = time.time()
	print 'link: {:.4f}s'.format(link_end - link_start)
	# links_all.append(links_cls)


	return links_all

def maxPath(dets_all,links_all):
    for cls_ind,links_cls in enumerate(links_all):
        max_begin=time.time()
        dets_cls=dets_all[cls_ind]
        while True:
            rootindex,maxpath,maxsum=findMaxPath(links_cls,dets_cls)
            if len(maxpath) <= 1:
                break
            rescore(dets_cls,rootindex,maxpath,maxsum)
            deleteLink(dets_cls,links_cls,rootindex,maxpath,C.overlap_thresh)
        max_end=time.time()
        print 'max path: {:.4f}s'.format(max_end - max_begin)

#base_model+model_alf
class Model_2step(Base_model):
	def name(self):
		return 'Model_2step'
	def initialize(self, opt):
		Base_model.initialize(self,opt)
		# specify the training details
		# 1st CPB loss
		self.cls_loss_r1 = []
		self.regr_loss_r1 = []
		#2nd CPB loss
		self.cls_loss_r2 = []
		self.regr_loss_r2 = []
		self.losses = np.zeros((self.epoch_length, 4))#4 losses
		self.optimizer = Adam(lr=opt.init_lr)
		print 'Initializing the {}'.format(self.name())

	def creat_model(self,opt,train_data, phase='train', wei_mov_ave = False):

		#通过创建base_model，指定anchor_scale以及ratio
		Base_model.create_base_model(self, opt,train_data, phase=phase, wei_mov_ave = wei_mov_ave)
		#return (1st CPB +2nd CPB) 的[P3,P4,P5,P6]预测值
		alf1, alf2 = model_alf.create_alf(self.base_layers, self.num_anchors, trainable=True, steps=2)
		#use weight move average
		if wei_mov_ave:
			alf1_tea, alf2_tea = model_alf.create_alf(self.base_layers_tea, self.num_anchors, trainable=True, steps=2)
			self.model_tea = Model(self.img_input, alf1_tea + alf2_tea)
		if phase=='train':
			#model_1st表示从输入到alf1的层的模型对象
			self.model_1st = Model(self.img_input, alf1)
			# model_2nd表示从头到alf2的层的模型对象
			self.model_2nd = Model(self.img_input, alf2)
			if self.num_gpus > 1:
				self.model_1st = ParallelModel(self.model_1st, int(self.num_gpus))
				self.model_2nd = ParallelModel(self.model_2nd, int(self.num_gpus))
			#在优化函数当中定义loss
			self.model_1st.compile(optimizer=self.optimizer, loss=[losses.cls_loss, losses.regr_loss],sample_weight_mode=None)
			self.model_2nd.compile(optimizer=self.optimizer, loss=[losses.cls_loss, losses.regr_loss],sample_weight_mode=None)
		self.model_all = Model(inputs=self.img_input, outputs=alf1+alf2)

	def train_model(self,opt, weight_path, out_path):
		self.model_all.load_weights(weight_path, by_name=True)
		print 'load weights from {}'.format(weight_path)
		iter_num = 0
		start_time = time.time()
		for epoch_num in range(self.num_epochs):
			progbar = generic_utils.Progbar(self.epoch_length)
			print('Epoch {}/{}'.format(epoch_num + 1 + self.add_epoch, self.num_epochs + self.add_epoch))
			while True:
				try:
					#yield np.copy(x_img_batch), [np.copy(y_cls_batch), np.copy(y_regr_batch)], np.copy(img_data_batch)
					#一共返回3个值:图片，Y,以及包含图片当中的所有的信息
					X, Y, img_data = next(self.data_gen_train)
					#calcuate 1st CPB loss
					loss_s1 = self.model_1st.train_on_batch(X, Y)
					self.losses[iter_num, 0] = loss_s1[1]
					self.losses[iter_num, 1] = loss_s1[2]
					pred1 = self.model_1st.predict_on_batch(X)
					#由pred1[1]生成y2的label,包含两个值，第一个是y_cls,第二个是y_reg
					Y2 = bbox_process.get_target_1st(self.anchors, pred1[1], img_data, opt,
													 igthre=opt.ig_overlap, posthre=opt.pos_overlap_step2,
													 negthre=opt.neg_overlap_step2)
					loss_s2 = self.model_2nd.train_on_batch(X, Y2)
					self.losses[iter_num, 2] = loss_s2[1]
					self.losses[iter_num, 3] = loss_s2[2]

					iter_num += 1
					if iter_num % 20 == 0:
						progbar.update(iter_num,
									   [('cls1', np.mean(self.losses[:iter_num, 0])),
										('regr1', np.mean(self.losses[:iter_num, 1])),
										('cls2', np.mean(self.losses[:iter_num, 2])),
										('regr2', np.mean(self.losses[:iter_num, 3]))])
					if iter_num == self.epoch_length:
						cls_loss1 = np.mean(self.losses[:, 0])
						regr_loss1 = np.mean(self.losses[:, 1])
						cls_loss2 = np.mean(self.losses[:, 2])
						regr_loss2 = np.mean(self.losses[:, 3])
						total_loss = cls_loss1 + regr_loss1 + cls_loss2 + regr_loss2

						self.total_loss_r.append(total_loss)
						self.cls_loss_r1.append(cls_loss1)
						self.regr_loss_r1.append(regr_loss1)
						self.cls_loss_r2.append(cls_loss2)
						self.regr_loss_r2.append(regr_loss2)

						print('Total loss: {}'.format(total_loss))
						print('Elapsed time: {}'.format(time.time() - start_time))

						iter_num = 0
						start_time = time.time()

						if total_loss < self.best_loss:
							print('Total loss decreased from {} to {}, saving weights'.format(self.best_loss, total_loss))
							self.best_loss = total_loss
						self.model_all.save_weights(
							os.path.join(out_path, 'resnet_e{}_l{}.hdf5'.format(epoch_num + 1 + self.add_epoch, total_loss)))
						break
				except Exception as e:
					print ('Exception: {}'.format(e))
					continue
			#To record the the train-loss result
			records = np.concatenate((np.asarray(self.total_loss_r).reshape((-1, 1)),
									  np.asarray(self.cls_loss_r1).reshape((-1, 1)),
									  np.asarray(self.regr_loss_r1).reshape((-1, 1)),
									  np.asarray(self.cls_loss_r2).reshape((-1, 1)),
									  np.asarray(self.regr_loss_r2).reshape((-1, 1))),
									 axis=-1)
			np.savetxt(os.path.join(out_path, 'records.txt'), np.array(records), fmt='%.6f')
		print('Training complete, exiting.')

	def train_model_wma(self,opt, weight_path, out_path):
		self.model_all.load_weights(weight_path, by_name=True)
		self.model_tea.load_weights(weight_path, by_name=True)
		print 'load weights from {}'.format(weight_path)
		iter_num = 0
		start_time = time.time()
		for epoch_num in range(self.num_epochs):
			progbar = generic_utils.Progbar(self.epoch_length)
			print('Epoch {}/{}'.format(epoch_num + 1 + self.add_epoch, self.num_epochs + self.add_epoch))
			while True:
				try:
					X, Y, img_data = next(self.data_gen_train)
					loss_s1 = self.model_1st.train_on_batch(X, Y)
					self.losses[iter_num, 0] = loss_s1[1]
					self.losses[iter_num, 1] = loss_s1[2]
					pred1 = self.model_1st.predict_on_batch(X)
					# Y2 = bbox_process.get_target_1st(self.anchors, pred1[1], img_data, opt,
					# 								 igthre=opt.ig_overlap, posthre=opt.pos_overlap_step2,
					# 								 negthre=opt.neg_overlap_step2)
					Y2 = bbox_process.get_target_1st_posfirst(self.anchors, pred1[1], img_data, opt,
													 igthre=opt.ig_overlap, posthre=opt.pos_overlap_step2,
													 negthre=opt.neg_overlap_step2)
					loss_s2 = self.model_2nd.train_on_batch(X, Y2)
					self.losses[iter_num, 2] = loss_s2[1]
					self.losses[iter_num, 3] = loss_s2[2]
					# apply weight moving average
					for l in self.model_tea.layers:
						weights_tea = l.get_weights()
						if len(weights_tea) > 0:
							weights_stu = self.model_all.get_layer(name=l.name).get_weights()
							weights_tea = [opt.alpha * w_tea + (1 - opt.alpha) * w_stu for (w_tea, w_stu) in
										   zip(weights_tea, weights_stu)]
							l.set_weights(weights_tea)

					iter_num += 1
					if iter_num % 20 == 0:
						progbar.update(iter_num,
									   [('cls1', np.mean(self.losses[:iter_num, 0])),
										('regr1', np.mean(self.losses[:iter_num, 1])),
										('cls2', np.mean(self.losses[:iter_num, 2])),
										('regr2', np.mean(self.losses[:iter_num, 3]))])
					if iter_num == self.epoch_length:
						cls_loss1 = np.mean(self.losses[:, 0])
						regr_loss1 = np.mean(self.losses[:, 1])
						cls_loss2 = np.mean(self.losses[:, 2])
						regr_loss2 = np.mean(self.losses[:, 3])
						total_loss = cls_loss1 + regr_loss1 + cls_loss2 + regr_loss2

						self.total_loss_r.append(total_loss)
						self.cls_loss_r1.append(cls_loss1)
						self.regr_loss_r1.append(regr_loss1)
						self.cls_loss_r2.append(cls_loss2)
						self.regr_loss_r2.append(regr_loss2)

						print('Total loss: {}'.format(total_loss))
						print('Elapsed time: {}'.format(time.time() - start_time))

						iter_num = 0
						start_time = time.time()

						if total_loss < self.best_loss:
							print('Total loss decreased from {} to {}, saving weights'.format(self.best_loss, total_loss))
							self.best_loss = total_loss
						self.model_tea.save_weights(
							os.path.join(out_path, 'resnet_e{}_l{}.hdf5'.format(epoch_num + 1 + self.add_epoch, total_loss)))
						break
				except Exception as e:
					print ('Exception: {}'.format(e))
					continue
			records = np.concatenate((np.asarray(self.total_loss_r).reshape((-1, 1)),
									  np.asarray(self.cls_loss_r1).reshape((-1, 1)),
									  np.asarray(self.regr_loss_r1).reshape((-1, 1)),
									  np.asarray(self.cls_loss_r2).reshape((-1, 1)),
									  np.asarray(self.regr_loss_r2).reshape((-1, 1))),
									 axis=-1)
			np.savetxt(os.path.join(out_path, 'records.txt'), np.array(records), fmt='%.6f')
		print('Training complete, exiting.')
	def test_model(self,opt, val_data, weight_path, out_path):
		self.model_all.load_weights(weight_path, by_name=True)
		print 'load weights from {}'.format(weight_path)
		res_all = []
		#用来保存预测结果
		res_file = os.path.join(out_path, 'val_det.txt')
		start_time = time.time()
		for f in range(len(val_data)):
			filepath = val_data[f]['filepath']
			# print (filepath)
			frame_number = f + 1
			img = cv2.imread(filepath)
			x_in = bbox_process.format_img(img, opt)
			#Y表示所有stage的分类以及回归的预测结果,2step返回4个值
			Y = self.model_all.predict(x_in)


			#Y[0],Y[1]表示 alf_1st的预测结果，Y[2],Y[3]表示alf+2nd的预测结果
			#Y[0]以及Y[2]表示cls_pred,Y[1]以及Y[3]表示reg_pred
			proposals = bbox_process.pred_pp_1st(self.anchors, Y[0], Y[1], opt)
			#bbx:[x1,y1,x2,y2]
			bbx, scores = bbox_process.pred_det(proposals, Y[2], Y[3], opt, step=2)
			f_res = np.repeat(frame_number, len(bbx), axis=0).reshape((-1, 1))
			#generate width and height
			bbx[:, [2, 3]] -= bbx[:, [0, 1]]
			res_all += np.concatenate((f_res, bbx, scores), axis=-1).tolist()
		np.savetxt(res_file, np.array(res_all), fmt='%.4f')
		print 'Test time: %.4f s' % (time.time() - start_time)

	# def test_model_bak(self,opt, val_data, weight_path, out_path):
	# 	self.model_all.load_weights(weight_path, by_name=True)
	# 	print 'load weights from {}'.format(weight_path)
	# 	res_all = []
	# 	res_file = os.path.join(out_path, 'my_val_det.txt')
	# 	start_time = time.time()
	# 	with open(res_file,'w')as ALF_f:
	# 		for f in range(len(val_data)):
	# 			filepath = val_data[f]['filepath']
	# 			# print (filepath)
	# 			filename=filepath.split('/')[-2:]
	# 			filename=filename[0]+'/'+filename[1]
	# 			filename=filename.split('.')[0]
	# 			frame_number = f + 1
	# 			img = cv2.imread(filepath)
	# 			x_in = bbox_process.format_img(img, opt)
	# 			Y = self.model_all.predict(x_in)
	# 			proposals = bbox_process.pred_pp_1st(self.anchors, Y[0], Y[1], opt)
	# 			bbx, scores = bbox_process.pred_det(proposals, Y[2], Y[3], opt, step=2)
	# 			# print (type(filename))
	# 			# print (type(scores))
    #
	# 			for i in range(bbx.shape[0]):
	# 				# print type(bbx[i, 3].item())
	# 				# print type(bbx[i,3].astype(int))
	# 				ALF_f.write('%s %f %d %d %d %d\n'%(filename,
	# 											   scores[i],
	# 											   int(bbx[i,0].item()),
	# 											   int(bbx[i, 1].item()),
	# 											   int(bbx[i,2].item()),
	# 											   int(
	# 												   bbx[i,3].item()
	# 											   )
	# 												   ))
    #
	# 			# f_res = np.repeat(filename, len(bbx), axis=0).reshape((-1, 1))
	# 			#generate width and height
	# 			# bbx[:, [2, 3]] -= bbx[:, [0, 1]]
	# 			# res_all += np.concatenate((f_res, bbx, scores), axis=-1).tolist()
	# 	# np.savetxt(res_file, np.array(res_all), fmt='%.4f')
	# 	print 'Test time: %.4f s' % (time.time() - start_time)
		
	def demo(self,opt, val_data, weight_path, out_path):
		self.model_all.load_weights(weight_path, by_name=True)
		print 'load weights from {}'.format(weight_path)
		for f in range(len(val_data)):
			img_name = os.path.join('data/examples/',val_data[f])
			if not img_name.lower().endswith(('.jpg', '.png')):
				continue
			# print(img_name)
			img = cv2.imread(img_name)
			x_in = bbox_process.format_img(img, opt)
			Y = self.model_all.predict(x_in)
			# print Y[2]
			proposals = bbox_process.pred_pp_1st(self.anchors, Y[0], Y[1], opt)
			bbx, scores = bbox_process.pred_det(proposals, Y[2], Y[3], opt, step=2)
			for ind in range(len(bbx)):
			    (x1, y1, x2, y2) = bbx[ind, :]
			    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.imwrite(os.path.join(out_path, val_data[f]),img)
	def demo_onepic(self,opt,img_path,weight_path,output_path):
		self.model_all.load_weights(weight_path, by_name=True)
		print 'load weights from {}'.format(weight_path)
		img=cv2.imread(img_path)
		x_in = bbox_process.format_img(img, opt)
		Y = self.model_all.predict(x_in)
		# print Y[2]
		proposals = bbox_process.pred_pp_1st(self.anchors, Y[0], Y[1], opt)
		bbx, scores = bbox_process.pred_det(proposals, Y[2], Y[3], opt, step=2)
		for ind in range(len(bbx)):
			(x1, y1, x2, y2) = bbx[ind, :]
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.imshow('det_result',img)
		cv2.waitKey(0)
		# img_file=output_path+'/'+img_path.split('/')[-1]
		# cv2.imwrite(filename=img_file,img=img)

	def read_video(self,video_path):
		cap=cv2.VideoCapture(video_path)
		ret,frame=cap.read()
		while True:
			ret, frame = cap.read()
			print type(frame)#<type 'numpy.ndarray'>
			cv2.imshow('MyWindow',frame)
			k = cv2.waitKey(1)
			if (k & 0xff == ord('q')):
				break
			# cap.release()
			# cv2.destroyAllWindows()

	def demo_video(self,opt,video_path,weight_path,output_path):
		self.model_all.load_weights(weight_path, by_name=True)
		print 'load weights from {}'.format(weight_path)
		print video_path
		timeF=3
		cap = cv2.VideoCapture(video_path)
		ret, frame = cap.read()
		#resize成为模型能读取的大小
		frame = cv2.resize(frame, (2048, 1024))
		#格式转化 ，减去mean
		x_in=bbox_process.format_img(frame, opt)


		if not cap.isOpened():
			raise IOError("Couldn't open webcam or video")

		# print (codec,size)
		codec = cv2.VideoWriter_fourcc(*'MJPG')
		size=((2048),(1024))
		out=cv2.VideoWriter(output_path,
							codec,
							15.0,#15 frame per second
							size)
		c=1
		dets=[]
		while cap.isOpened():
			det=[]
			ret,frame = cap.read()
			if frame is not None:
				if (c%timeF==0):
					frame = cv2.resize(frame, (2048, 1024))
					x_in = bbox_process.format_img(frame, opt)
					Y=self.model_all.predict(x_in)
					proposals = bbox_process.pred_pp_1st(self.anchors, Y[0], Y[1], opt)
					bbx, scores = bbox_process.pred_det(proposals, Y[2], Y[3], opt, step=2)
					# dets=np.hstack((bbx,scores[:,np.newaxis])).astype(float64)
					for ind in range(len(bbx)):
						(x1, y1, x2, y2) = bbx[ind, :]
						# print (x1,y1,x2,y2)


						# if (x2-x1<100)&(y2-y1<350)&(x2-x1>25)&(y2-y1>150):
						# 	cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
						# 	det.append([x1,y1,x2,y2])

						# if (x2-x1<100)&(y2-y1<350)&(x2-x1>25)&(y2-y1>150):
						cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
						# det.append([x1,y1,x2,y2])

					# print type(frame)
					dets.append(det)
					cv2.imshow('detection',frame)
					out.write(frame)
					k = cv2.waitKey(1)
					if (k & 0xff == ord('q')):
						break
				c=c+1
			else:
				break
		# 调用dets里面保存的所有帧的检测信息，进行帧与帧之间的关联
		# links = createLinks(dets)
		# maxPath(dets, links)
		cap.release()
		cv2.destroyAllWindows()













