import argparse
import numpy as np
import os
from tracker import Tracker
import math
import json

parser = argparse.ArgumentParser(description='Test some videos.')

parser.add_argument('--img_path', type = str, default = 'sample_data/ball1_img/')
parser.add_argument('--gt_bbox_path', type = str, default = 'sample_data/ball1_gt/')
parser.add_argument('--object_name', type = str, default = 'sports ball')
parser.add_argument('--video_name', type = str, default = 'ball1')
parser.add_argument('--auxiliary_tracker', type = str, default = 'SIAM', choices=['SIAM', 'KCF'])
parser.add_argument('--long_term', default= True, action='store_false')
parser.add_argument('--exp_name', default= None, type = str)
# parser.add_argument('--gpu_id', type = str)
# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

def main():
	args = parser.parse_args()

	file1 = open(args.gt_bbox_path+"groundtruth.txt","r+")

	bbox_init  = file1.readline().strip().split(',')
	bbox_init = [int(float(val)) for val in bbox_init]

	tracker = Tracker(img_list_dir= args.img_path, 
	             init_gt_bbox = bbox_init, 
	             object_name= args.object_name,
	             auxiliary_tracker_type = args.auxiliary_tracker,
	             short_term = args.long_term)

	#results
	bbox = tracker.bbox_results

	#for each frame whether TDIOT os KCF/SIAM has been an active
	tracker_name = tracker.frame_tracker

	if args.exp_name is None:
	    args.exp_name = f'VideoName_{args.video_name}_ObjNmae_{args.object_name}_{args.auxiliary_tracker}_shortTerm_{args.long_term}' 

	with open(f'output/{args.exp_name}_args.json', 'wt') as f:
	    json.dump(vars(args), f, indent=2)

	with open( os.path.join('output', args.exp_name+'.txt'), 'a+') as f:
	    for i in range(bbox.shape[0]):
	        res_bbox_frame = "{}\t{}\t{}\t{}\n".format(format(bbox[i,0],'.4f'), format(bbox[i,1],'.4f'), format(bbox[i,2],'.4f'), format(bbox[i,3],'.4f'))
	        f.write(res_bbox_frame)

if __name__ == "__main__":
    main()
