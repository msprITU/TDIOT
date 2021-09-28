import numpy as np
import cv2

def convert_seq_to_video(filename,img_seqs):
    img_dim = img_seqs[0].shape
    size = (img_dim[1],img_dim[0])
    
    out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_seqs)):
        out.write(img_seqs[i])
    out.release()