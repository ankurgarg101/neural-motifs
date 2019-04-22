"""
Visualization script. I used this to create the figures in the paper.

WARNING: I haven't tested this in a while. It's possible that some later features I added break things here, but hopefully there should be easy fixes. I'm uploading this in the off chance it might help someone. If you get it to work, let me know (and also send a PR with bugs/etc)
"""

from dataloaders.visual_genome import VGDataLoader, VG
from lib.rel_model import RelModel
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
from lib.fpn.box_utils import bbox_overlaps
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import os
from functools import reduce
import json
from collections import OrderedDict

conf = ModelConfig()
train, val, test = VG.splits(num_val_im=conf.val_size)
if conf.test:
	val = test

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
											   batch_size=conf.batch_size,
											   num_workers=conf.num_workers,
											   num_gpus=conf.num_gpus)

detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
					num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
					use_resnet=conf.use_resnet, order=conf.order,
					nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
					use_proposals=conf.use_proposals,
					pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
					pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
					pooling_dim=conf.pooling_dim,
					rec_dropout=conf.rec_dropout,
					use_bias=conf.use_bias,
					use_tanh=conf.use_tanh,
					limit_vision=conf.limit_vision
					)
detector.cuda()
ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])

def val_epoch():
	detector.eval()
	sg_dict = OrderedDict()
	object_counter = 0
	for val_b, batch in enumerate(tqdm(val_loader)):
		fn = os.path.split(val.filenames[val_b])[1][:-4]
		height, width, factor = batch[0][1][0]
		height, width = height/factor, width/factor
		sg_dict[fn] = OrderedDict()
		sg_dict[fn]['width'] = width
		sg_dict[fn]['height'] = height
		sg_dict[fn]['objects'], object_counter = val_batch(conf.num_gpus * val_b, batch, factor, object_counter)
	return sg_dict

def val_batch(batch_num, b, factor, object_counter, pred_thresh=0.5, obj_thresh=0.5):
	det_res = detector[b]
	assert conf.num_gpus == 1
	boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = det_res
	assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)
	
	#Convert boxes to actual image size
	boxes_i /= factor
	boxes_i[:, 2] -= boxes_i[:, 0] 
	boxes_i[:, 3] -= boxes_i[:, 1] 
	
	#Get the actual prediction of predicates classification
	pred_class_i = np.argmax(pred_scores_i, axis=1)
	pred_class_score_i = np.max(pred_scores_i, axis=1)

	#Do confidence threshold of object
	obj_filter = np.where(obj_scores_i>=obj_thresh)[0]
	boxes_i = boxes_i[obj_filter]
	objs_i = objs_i[obj_filter]
	obj_scores_i = obj_scores_i[obj_filter]

	#Do confidence threshold of predicates
	pred_filter1 = np.where(pred_class_score_i>=pred_thresh)[0]
	rels_i = rels_i[pred_filter1]
	pred_scores_i = pred_scores_i[pred_filter1]
	pred_class_i = pred_class_i[pred_filter1]
	pred_class_score_i = pred_class_score_i[pred_filter1]

	#Remove the relations of the removed objects
	pred_filter2 = np.where(np.isin(rels_i[:,0], obj_filter) & np.isin(rels_i[:,1], obj_filter))[0]
	rels_i = rels_i[pred_filter2]
	pred_scores_i = pred_scores_i[pred_filter2]
	pred_class_i = pred_class_i[pred_filter2]
	pred_class_score_i = pred_class_score_i[pred_filter2]

	objects = {}
	start_id = object_counter
	obj_filter = list(obj_filter)
	for i in range(len(obj_filter)):
		obj_id = str(object_counter)
		objects[obj_id] = OrderedDict()
		objects[obj_id]['name'] = train.ind_to_classes[objs_i[i]]
		objects[obj_id]['x'] = int(boxes_i[i,0])
		objects[obj_id]['y'] = int(boxes_i[i,1])
		objects[obj_id]['w'] = int(boxes_i[i,2])
		objects[obj_id]['h'] = int(boxes_i[i,3])
		objects[obj_id]['attributes'] = []
		objects[obj_id]['relations'] = []
		#Filter relations of same object as subject
		filtered_rels_i = np.where(rels_i[:,0]==obj_filter[i])[0]
		for rel_id in filtered_rels_i:
			sub, obj = rels_i[rel_id]
			obj = obj_filter.index(obj) # because of filtering of objects, the indices have changed
			rel_dict = OrderedDict()
			rel_dict['name'] = train.ind_to_predicates[pred_class_i[rel_id]]
			rel_dict['object'] = str(start_id + obj)
			objects[obj_id]['relations'].append(rel_dict)
		object_counter += 1
	return objects, object_counter

scenegraph = val_epoch()

pathname = 'qualitative_sg'
if not os.path.exists(pathname):
	os.makedirs(pathname)

pathname = os.path.join(pathname,'sg_{}.json'.format(conf.mode))
with open(pathname,'w') as f:
	json.dump(scenegraph, f)
