import os
import json
import math
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf 
from skimage.transform import resize
import skimage.io as io
import random
from pycocotools.coco import COCO 

class Coco_datagen:
	def __init__(self,data_dir,anno_type,model_input_shape,model_output_shape,batch_size_select=64):
		self.img_dir = data_dir
		self.bbox = []
		self.kps_and_valid = []
		self.kps = []
		self.scaled_kps = []
		self.valids = []
		self.img_ids = []
		self.imgs = []
		self.start_idx = []
		self.end_idx = []
		self.input_shape = model_input_shape
		self.output_shape = model_output_shape
		self.n_imgs = None
		self.batch_size = batch_size_select
		self.n_batchs = None
		self.sig = 2

		dataDir = '.'
		dataType = anno_type
		annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
		self.coco_kps = COCO(annFile)

		self.annIds = self.coco_kps.getAnnIds()
		self.anns = self.coco_kps.loadAnns(self.annIds)

		self.img_ids = [ann['image_id'] for ann in self.anns]
		self.n_imgs = len(self.img_ids)
		self.bbox = [ann['bbox'] for ann in self.anns]
		self.kps_and_valid = [ann['keypoints'] for ann in self.anns]

		self.start_idx, self.end_idx, self.n_batchs = self.get_start_end_idx()

		self.imgs = self.coco_kps.loadImgs(self.img_ids)
		self.kps,self.valids = self.get_target_valid_joint()
		self.scale_bbox_to_43()
		self.convert_kps_to_local()
		self.scale_kps_to_output_size()

	def get_start_end_idx(self):
		max_idx = self.n_imgs
		temp_batch_size = self.batch_size
		l = list(range(max_idx))
		temp_start_idx = l[0::temp_batch_size]
		def add_batch_size(num,max_id=max_idx,bz=temp_batch_size):
			return min(num+bz,max_id)
		temp_end_idx = list(map(add_batch_size,temp_start_idx))
		temp_n_batchs = len(temp_start_idx)

		return temp_start_idx, temp_end_idx, temp_n_batchs
	def get_target_valid_joint(self):
		temp_anno_kp_valid = self.kps_and_valid
		temp_kps = []
		temp_valids = []
		for temp_anno_kp in temp_anno_kp_valid:
			
			temp_x = np.array(temp_anno_kp[0::3])
			temp_x = np.delete(temp_x,[1,2])

			temp_y = np.array(temp_anno_kp[1::3])
			temp_y = np.delete(temp_y,[1,2])

			temp_valid = np.array(temp_anno_kp[2::3])
			temp_valid = np.delete(temp_valid,[1,2])

			temp_valid = temp_valid > 0
			temp_valid = temp_valid.astype('float32')
			temp_target_coord = np.stack([temp_x,temp_y],axis=1)
			temp_target_coord = temp_target_coord.astype('float32')

			temp_kps.append(temp_target_coord)
			temp_valids.append(temp_valid)
			

		return temp_kps,temp_valids

	def scale_bbox_to_43(self):
		temp_bbox = self.bbox
		temp_imgs = self.imgs

		for i in range(len(temp_imgs)):
			img_h = temp_imgs[i]['height']
			img_w = temp_imgs[i]['width']

			i_bbox = temp_bbox[i]
			bbox_x, bbox_y, bbox_w, bbox_h = i_bbox

			to_check = 0.75*bbox_h
			if to_check >= bbox_w:
				add_x = True
			else:
				add_x = False

			if add_x:
				new_bbox_h = bbox_h
				new_bbox_w = 0.75*bbox_h
				diff = new_bbox_w - bbox_w
				new_bbox_y = bbox_y
				new_bbox_x = bbox_x - 0.5*diff
				#check if in image
				if new_bbox_x < 0:
					new_bbox_x = 0
				if new_bbox_x+new_bbox_w >= img_w:
					new_bbox_x = img_w - new_bbox_w - 1
			else:
				new_bbox_w = bbox_w
				new_bbox_h = 4.0/3.0 * bbox_w
				diff = new_bbox_h - bbox_h
				new_bbox_x = bbox_x
				new_bbox_y = bbox_y - 0.5 * diff
				if new_bbox_y < 0:
					new_bbox_y = 0
				if new_bbox_y+new_bbox_h >= img_h:
					new_bbox_y = img_h - new_bbox_h - 1
			temp_new_bbox = [new_bbox_x,new_bbox_y,new_bbox_w,new_bbox_h]
			temp_bbox[i] = temp_new_bbox
		self.bbox = temp_bbox
	
	def convert_kps_to_local(self):
		temp_bbox = self.bbox
		temp_kps = self.kps
		for i in range(len(temp_kps)):
			i_bbox = temp_bbox[i]
			i_kps = temp_kps[i]
			i_origin = np.array([i_bbox[0], i_bbox[1]])
			for j in range(len(i_kps)):
				if i_kps[j,0] != 0:
					i_kps[j,:] = i_kps[j,:] - i_origin
			temp_kps[i] = i_kps
		self.kps = temp_kps

	def scale_kps_to_output_size(self):
		temp_bbox = self.bbox
		temp_kps = self.kps
		temp_output_shape = self.output_shape
		for i in range(len(temp_bbox)):
			i_bbox = temp_bbox[i]
			i_kps = temp_kps[i]
			if i_bbox[2] == 0 or i_bbox[3] == 0:
				i_bbox[3] = 1
			scale_x = temp_output_shape[1] / i_bbox[2]
			scale_y = temp_output_shape[0] / i_bbox[3]
			temp_scale = np.array([scale_x, scale_y])
			for j in range(len(i_kps)):
				temp_value = i_kps[j,:]
				i_kps[j,:] = np.multiply(temp_value,temp_scale)
			temp_kps[i] = i_kps
		self.scaled_kps = temp_kps

	def shuffle_order(self):
		temp_bbox = self.bbox
		temp_kps = self.kps
		temp_scaled_kps = self.scaled_kps
		temp_img_ids = self.img_ids
		temp_imgs = self.imgs
		temp_valids = self.valids
		to_shuffle = list(zip(temp_bbox,temp_kps,temp_scaled_kps,temp_imgs,temp_img_ids,temp_valids))
		random.shuffle(to_shuffle)
		temp_bbox,temp_kps,temp_scaled_kps,temp_imgs,temp_img_ids,temp_valids = zip(*to_shuffle)

		self.bbox = temp_bbox
		self.kps = temp_kps
		self.scaled_kps = temp_scaled_kps
		self.imgs = temp_imgs
		self.img_ids = temp_img_ids
		self.valids = temp_valids

	def render_gaussian_heatmap(self,input_kps,input_valids,sigma):
		r_output_shape = self.output_shape
		x = [i for i in range(r_output_shape[1])]
		y = [i for i in range(r_output_shape[0])]
		xx,yy = tf.meshgrid(x,y)
		xx = tf.reshape(tf.cast(xx,tf.float32),(1,*r_output_shape,1))
		yy = tf.reshape(tf.cast(yy,tf.float32),(1,*r_output_shape,1))

		input_kps_float = input_kps.astype(np.float64)

		x = tf.floor(tf.reshape(input_kps_float[:,0],[-1,1,1,15])+ 0.5 )
		y = tf.floor(tf.reshape(input_kps_float[:,1],[-1,1,1,15])+ 0.5 )
		x = tf.cast(x,tf.float32)
		y = tf.cast(y,tf.float32)
		temp_heatmap = tf.exp(-(((xx-x)/tf.cast(sigma,tf.float32))**2)/tf.cast(2,tf.float32) - (((yy-y)/tf.cast(sigma,tf.float32))**2)/tf.cast(2,tf.float32))
		temp_heatmap = temp_heatmap * 255.
		temp_heatmap = temp_heatmap.numpy()
		temp_heatmap = np.reshape(temp_heatmap,(*r_output_shape,15))

		for ii in range(len(input_valids)):
			if input_valids[ii] <= 0:
				temp_heatmap[:,:,ii] = np.zeros(r_output_shape)
		return temp_heatmap

	def gen_batch(self,batch_order):
		batch_imgs = []
		batch_heatmaps = []
		batch_valids = []
		b_start = self.start_idx[batch_order]
		b_end = self.end_idx[batch_order]
		temp_output_shape = self.output_shape
		temp_input_shape = self.input_shape
		temp_valids = self.valids
		# temp_img_ids = self.img_ids
		temp_imgs = self.imgs
		temp_bbox = self.bbox
		temp_kps = self.scaled_kps
		temp_imgdir = self.img_dir
		# temp_dict = self.id_to_file_dict
		for i in range(b_start,b_end):
			#valid
			i_valid = temp_valids[i]
			i_ones = np.ones((*temp_output_shape,15),dtype = np.float32)
			o_valid = i_ones*i_valid

			#heatmap
			i_kp = temp_kps[i]
			o_heatmap = self.render_gaussian_heatmap(i_kp,i_valid,self.sig)
			#imgs
			i_img = temp_imgs[i]
			o_img = io.imread('./'+ temp_imgdir + '/' + i_img['file_name'])
			i_bbox = temp_bbox[i]
			if o_img.shape[0] == 0 or o_img.shape[1]  == 0 or o_img.ndim < 3:
				continue
			o_crop = o_img[int(i_bbox[1]):int(i_bbox[1]+i_bbox[3]),int(i_bbox[0]):int(i_bbox[0]+i_bbox[2]),:]
			# detect empthy image
			if o_crop.shape[0] == 0 or o_crop.shape[1]  == 0 or o_crop.shape[2] == 0:
				continue
			o_crop = resize(o_crop,temp_input_shape)
			o_crop = o_crop.astype('float32')

			batch_imgs.append(o_crop)
			batch_heatmaps.append(o_heatmap)
			batch_valids.append(o_valid)
		batch_imgs = np.array(batch_imgs)
		batch_heatmaps = np.array(batch_heatmaps)
		batch_valids = np.array(batch_valids)

		return batch_imgs, batch_heatmaps, batch_valids