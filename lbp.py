import numpy as np
import cv2
#from time import perf_counter

def lbp_decimal_4(img):

    image_base_ = img[1:-1,1:-1] 
    
    image_up = img[:-2,1:-1]
    image_down = img[2:,1:-1]
    
    image_left = img[1:-1,:-2]
    image_right = img[1:-1:,2:]
    
     
    pow_1 = np.power(2,0)*1.0*(image_up >= image_base_)
    pow_2 = np.power(2,1)*1.0*(image_down >= image_base_)
    pow_3 = np.power(2,2)*1.0*(image_left >= image_base_)
    pow_4 = np.power(2,3)*1.0*(image_right >= image_base_)
    
    lbp_decimal = pow_1 + pow_2 + pow_3 + pow_4
    
    decimal_flatten = lbp_decimal.flatten()
    
    hist_lbp,bins = np.histogram(decimal_flatten, bins=np.arange(16))
    
    return hist_lbp,bins

def lbp_decimal_8(img, hist_conc= False):

    image_base_ = img[1:-1,1:-1] 
    
    image_up = img[:-2,1:-1]
    image_down = img[2:,1:-1]
    
    image_left = img[1:-1,:-2]
    image_right = img[1:-1:,2:]
    
    image_diag_up_left = img[:-2,:-2]
    image_diag_up_right = img[:-2,2:]
    
    image_diag_down_left = img[2:,:-2]
    image_diag_down_right = img[2:,2:]
    
    pow_1 = np.power(2,0)*1.0*(image_up >= image_base_)
    pow_2 = np.power(2,1)*1.0*(image_diag_up_right >= image_base_)
    pow_3 = np.power(2,2)*1.0*(image_right >= image_base_)
    pow_4 = np.power(2,3)*1.0*(image_diag_down_right >= image_base_)
    pow_5 = np.power(2,4)*1.0*(image_down >= image_base_)
    pow_6 = np.power(2,5)*1.0*(image_diag_down_left >= image_base_)
    pow_7 = np.power(2,6)*1.0*(image_left >= image_base_)
    pow_8 = np.power(2,7)*1.0*(image_diag_up_left >= image_base_)
    
    lbp_decimal = pow_1 + pow_2 + pow_3 + pow_4 + pow_5 + pow_6 + pow_7 + pow_8

    if hist_conc==True:
        decimal_flatten_1 = lbp_decimal[:,:,0].flatten()
        decimal_flatten_2 = lbp_decimal[:,:,1].flatten()
        decimal_flatten_3 = lbp_decimal[:,:,2].flatten() 
        
        hist_lbp_1, bins = np.histogram(decimal_flatten_1, bins=np.arange(256))
        hist_lbp_2, bins = np.histogram(decimal_flatten_2, bins=np.arange(256))
        hist_lbp_3, bins = np.histogram(decimal_flatten_3, bins=np.arange(256))

        hist_lbp = np.concatenate((hist_lbp_1,hist_lbp_2,hist_lbp_3), axis =0, out= None)

    else:
        decimal_flatten = lbp_decimal.flatten()
        hist_lbp,bins = np.histogram(decimal_flatten, bins=np.arange(256))

    return hist_lbp,bins

def lbp_decimal_8_loop(img):
    
    row, col = img.shape
    lbp_image = np.empty((row-2, col-2))
    
    for i in range(1,row-1):
        for j in range(1,col-1):
            
            pixel_val= img[i,j]
            lbp_val = np.power(2,0)*(img[i-1,j] > pixel_val ) + \
                      np.power(2,1)*(img[i-1,j+1] > pixel_val) + \
                      np.power(2,2)*(img[i,j+1] > pixel_val) + \
                      np.power(2,3)*(img[i+1,j+1] > pixel_val) + \
                      np.power(2,4)*(img[i+1,j] > pixel_val) + \
                      np.power(2,5)*(img[i+1,j-1] > pixel_val) + \
                      np.power(2,6)*(img[i,j-1] > pixel_val) + \
                      np.power(2,7)*(img[i-1,j-1] > pixel_val) 
                      
            lbp_image[i-1, j-1]= lbp_val
    
    decimal_flatten = lbp_image.flatten()
    
    hist_lbp,bins = np.histogram(decimal_flatten, bins=np.arange(256))

    return hist_lbp,bins
        
def chi_square_dist(arr1,arr2, normalize=False):
    if normalize== True:
        arr1= arr1/(np.sum(arr1)/3.0)
        arr2= arr2/(np.sum(arr2)/3.0)
    
        
    # coeff = np.sum(arr1*1.0)/np.sum(arr2)
    # arr2 = coeff*arr2

    num = np.sum((arr1-arr2)**2)
    dom = np.sum(arr1+arr2)

    dist = num/dom
    
    # print("\n\nN D DIST : ",num, "\t", dom,"\t",dist)
    return dist

def KL_dist(arr1,arr2):
    # arr1= arr1/(np.sum(arr1)/3.0)
    # arr2= arr2/(np.sum(arr2)/3.0)

    # # print("\n\n")
    # # print(arr1)
    # # print("\n\n")
    # # print(arr2)

    # sum_kl = np.sum(np.where(arr1 != 0, arr1 * np.log(arr1 / arr2), 0))
    return -1

def resize_bbox(src, ref_w, ref_h):

    h_src, w_src, d = src.shape

    fx = ref_w/w_src
    fy = ref_h/h_src
    src_resize = cv2.resize(src,(0,0), fx=fx, fy=fy, interpolation= cv2.INTER_LINEAR)
    return src_resize

def L2_dist(arr1,arr2, normalize=False):
    if normalize== False:
        arr1= arr1/(np.sum(arr1)/3.0)
        arr2= arr2/(np.sum(arr2)/3.0)
    
        dist = np.sum((arr1-arr2)**2)

    dist = dist
    
    # print("\n\nN D DIST : ",num, "\t", dom,"\t",dist)
    return dist

