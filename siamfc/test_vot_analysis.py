from __future__ import absolute_import

import numpy as np
import os
import ops
import cv2
#from got10k.experiments import ExperimentOTB

from siamfc import TrackerSiamFC

def read_txt(path_file):
    mat = []
    with open(path_file) as f:
        lines = f.readlines()
    for line in lines:
        if line.find(" ")>0:
#            line.replace("\n","")
            values_str = line.split(" ")
            values_str_1 = [float(val.replace("\n",""))for val in values_str]
            mat.append(values_str_1)
    
    
    mat = np.array(mat)
    mat= np.round(mat)
    return mat
    

if __name__ == '__main__':
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    
    gt_path = "C:/Users/llukm/Desktop/vot2016_gt/"
    gt_files = []
    with os.scandir(gt_path) as entries:
        for entry in entries:
#            print(entry.name)
            gt_files.append(gt_path+entry.name)
    
#    gt = read_txt("C:/Users/llukm/Desktop/vot2016_gt/duzgun_gt(vot2016)_basketball.txt")
#    
#
    path_imgs = 'C:/Users/llukm/Desktop/TDOT/data_VOT2018/'
    img_file_names = []
    video_name = []
    with os.scandir(path_imgs) as entries:
        for entry in entries:
#            print(entry.name)
            video_name.append(entry.name)
            img_file_names.append(path_imgs+entry.name+"/")
    
    for i in range(len(gt_files)): #len(gt_files)
        
        img_files= []
        with os.scandir(img_file_names[i]) as entries:
            for entry in entries:

                img_files.append(img_file_names[i]+entry.name)
        
        
        if i ==2:
            
            print(video_name[i])
            
            gt = read_txt(gt_files[i])
#            boxes, times = tracker.track(img_files[:10], gt[0, :], visualize=False)
            
            for f, img_file in enumerate(img_files[:100]):
                img = ops.read_image(img_file)
    
         
                if f == 0:
                    tracker.init(img, gt[0, :])
                else:
                    boxes = tracker.update(img)
                    print(boxes)
                    ops.show_image(img, boxes)
                    
#                    tracker.re_init_params(tracker.center, tracker.target_sz)
                    tracker.re_init_params(np.array([30.0,40.5], dtype = np.float32), np.array([20.0,45.0], dtype = np.float32))
                    
        cv2.destroyAllWindows()
                
        

        
        
        
        