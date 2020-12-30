# TDIOT
Video sequences uploaded to the following link demonstrate the short term and long term performance achieved by TDIOT on VOT2016, VOT2018, and VOT-LT2018.
https://youtube.com/playlist?list=PLMzonaXew-54_VaEJNn4TdrZkWpPeE4eW

## Setup (with Anaconda)

### Install dependencies

The code is tested with CUDA 10.1 on Linux. 
Note that the code also can run without gpu. 

Required packages:

 * tensorflow (gpu-optional) = 1.3

 * Keras = 2.0.8
 
 * PyTroch (CPU) = 1.5
 
 * scikit-image
 
 * opencv
 
 * got10k
 
 * ipython
 
 * seaborn
 
 * h5py=2.10.0
 

## To run the code

You can run the code for two differeent scanerios: short term and long term.

After you run the code you will get results in text file format.
In each row you will get (x,y) - top lef corner and (w,h)-width and height of the object. 

### Long Term (LT) Tracking
 Run the following 

```
main_demo_lt.py
```

### Short Term (ST) Tracking
 Run the following 

```
main_demo_st.py
```

The final version of the code will be released soon.
If you would like to download the beta version send an e-mail to llukmancerkezi@gmail.com  and also CC gunselb@itu.edu.tr

## Credits

Mask R-CNN and pretrained weights: [Here](https://github.com/matterport/Mask_RCNN)

KCF code is based on: [Here](https://github.com/fengyang95/pyCFTrackers)

SiamFC code and the weights are based on :[Here](https://github.com/huanglianghua/siamfc-pytorch)

