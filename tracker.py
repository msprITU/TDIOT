import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import argparse
import cv2
import shutil
import time

from mrcnn import utils
import mrcnn.model as modellib
import mrcnn.visualize
from mrcnn.config import Config

from kcf.kcf_particle_full_version_1 import KCF

from lbp import lbp_decimal_8
from lbp import chi_square_dist
from lbp import L2_dist
from lbp import resize_bbox

import keras.backend as K
import tensorflow as tf

from siamfc.siamfc_ import TrackerSiamFC

gpu_options = tf.GPUOptions(allow_growth=True)
# gpu_options = tf.GPUOptions(allocator_type = 'BFC')
config_keras = tf.ConfigProto(gpu_options=gpu_options)
K.set_session(tf.Session(config=config_keras))

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# INTERNAL PARAMETERS

PARTICLE_CNT = 200
ARG_MODE = "extension"
ARG_COCO_MODEL_PATH = "mrcnn//mask_rcnn_coco.h5"

GAUSS_STDEV_COORD_INIT = 5
GAUSS_STDEV_SCALE_INIT = 5
GAUSS_STDEV_COORD = GAUSS_STDEV_COORD_INIT
GAUSS_STDEV_SCALE = GAUSS_STDEV_SCALE_INIT
STDEV_COORD_MISS_MULTIPLIER = 3  # NEW_GAUSS_STDEV_COORD = OLD_GAUSS_STDEV_COORD * STDEV_COORD_MISS_MULTIPLIER
STDEV_SCALE_MISS_MULTIPLIER = 1  # NEW_GAUSS_STDEV_SCALE = OLD_GGAUSS_STDEV_SCALE * STDEV_SCALE_MISS_MULTIPLIER


class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    global ARG_MODE
    NAME = "coco_evaluation"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # NUM_CLASSES = 1 + 1
    NUM_CLASSES = 80 + 1
    DETECTION_MIN_CONFIDENCE = 0.0
    DETECTION_NMS_THRESHOLD = 0.7
    FILTER_BACKGROUND = True

    if ARG_MODE == "extension" or "rpn":
        RPN_MODE = 0  # Mode 1
        # POST_PS_ROIS_INFERENCE = PARTICLE_COUNT
        PARTICLE_COUNT = PARTICLE_CNT
        RPN_OUTPUT_COUNT = 1000 + PARTICLE_COUNT
        DETECTION_MAX_INSTANCES = 400
        # POST_PS_ROIS_INFERENCE = 1000
        # DETECTION_MAX_INSTANCES = 1000
    elif ARG_MODE == "inference":
        POST_PS_ROIS_INFERENCE = 1000
        DETECTION_MAX_INSTANCES = 1000

    INIT_BN_BACKBONE = True
    INIT_GN_BACKBONE = False
    INIT_BN_HEAD = True
    INIT_GN_HEAD = False


def particle_mask_normalize(particles, first_img, config):
    first_img = skimage.io.imread(first_img)
    dim_y, dim_x, _ = first_img.shape

    for i in range(0, particles.shape[0]):
        # print("particle [{}] before: {}".format(i,particles[i]))
        x, y, w, h = particles[i]
        particles[i, 0] = y / dim_y
        particles[i, 1] = x / dim_x
        particles[i, 2] = (y + h) / dim_y
        particles[i, 3] = (x + w) / dim_x
        # print("particle [{}] after: {}".format(i,particles[i]))

    particles_np = np.array(particles).astype(np.float64)

    # Get the scale and padding parameters by using resize_image.
    _, _, scale, pad, _ = utils.resize_image(first_img,
                                             min_dim=config.IMAGE_MIN_DIM,
                                             max_dim=config.IMAGE_MAX_DIM,
                                             min_scale=config.IMAGE_MIN_SCALE,
                                             mode="square")

    # Roughly calculate padding across different axises.
    aver_pad_y = (pad[0][0] + pad[0][1]) / 2
    aver_pad_x = (pad[1][0] + pad[1][1]) / 2

    particles_np *= np.array((dim_y, dim_x, dim_y, dim_x))
    particles_np = (particles_np * scale) + np.array((aver_pad_y, aver_pad_x, aver_pad_y, aver_pad_x))
    particles_np /= np.array(
        (1024, 1024, 1024, 1024))  ## ozgun soru: bunun config.IMAGE_MAX_DIM vs olması gerekmiyor muydu?
    particles_np = particles_np.reshape((-1, config.PARTICLE_COUNT, 4))
    return particles_np


def are_corners_inimage(image, x, y, w, h):
    overflow = None
    row, col, d = image.shape

    if y < 0:
        overflow = 'U'
    elif y + h > row:
        overflow = 'D'

    elif x < 0:
        overflow = 'L'
    elif x + w > col:
        overflow = 'R'

    return overflow


def crop_image(image, x, y, w, h):
    overflow = are_corners_inimage(image, x, y, w, h)

    if y < 0:
        y = 0
    if x < 0:
        x = 0

    img = image[y:y + h, x:x + w, :]
    if img.size == 0:
        return None, overflow
    else:
        return img, overflow


class Tracker():
    def __init__(self,
                 img_list_dir,
                 init_gt_bbox,
                 object_name,
                 auxiliary_tracker_type="KCF",
                 short_term=True):

        self.img_list_dir = img_list_dir

        self.object_name = object_name
        self.auxiliary_tracker_type = auxiliary_tracker_type
        self.short_term = short_term

        self.target_id = self.find_target_id()
        assert self.target_id != -1, "object name is not found in class name list !"

        #        assert self.auxilary_tracker_type == "KCF" or  self.auxilary_tracker_type == "SIAM", "Please choose valid auxilary tracker KCF/SIAM!"


        if len(init_gt_bbox) == 8:
            self.init_gt_bbox = self.convert_coordinates(init_gt_bbox)
        else:
            self.init_gt_bbox = init_gt_bbox

        # Read all video frames
        self.image_ids = self.read_image_file_names(self.img_list_dir)
        self.nb_frames = len(self.image_ids)

        self.bbox_results = np.zeros((self.nb_frames - 1, 4))
        self.frame_tracker = []

        self.auxiliary_tracker = self._choose_auxiliary_tracker()

        init_image = skimage.io.imread(os.path.join(self.img_list_dir, self.image_ids[0]))
        dims = init_image.shape
        canvas_shape_x_y_format = [dims[1], dims[0]]

        #        self.particles = np.zeros((1, PARTICLE_COUNT, 4))
        self.particles = utils.sample_bbox(seed=self.init_gt_bbox, stdev_coord=GAUSS_STDEV_COORD,
                                           stdev_scale=GAUSS_STDEV_SCALE, count=PARTICLE_CNT,
                                           canvas_shape=canvas_shape_x_y_format)

        if self.short_term == True:
            self.short_term_tracking()
        else:
            self.long_term_tracking()

    def _choose_auxiliary_tracker(self):
        if self.auxiliary_tracker_type == "KCF":
            new_auxiliary_tracker = KCF(padding=1, features='color', interp_factor=0.00)
        else:
            # make it local !!!!
            # siam_net_path = 'C:/Users/llukm/Desktop/TDOT/MaskRCNN_Feedback/siamfc/pretrained_siamfc/siamfc_alexnet_e50.pth'
            siam_net_path = 'siamfc/pretrained_siamfc/siamfc_alexnet_e50.pth'
            new_auxiliary_tracker = TrackerSiamFC(net_path=siam_net_path)

        return new_auxiliary_tracker

    def long_term_tracking(self):

        print("\n\n\n LONG TERM TRACKING\n\n")
        print(self.auxiliary_tracker_type)

        init_image = skimage.io.imread(os.path.join(self.img_list_dir, self.image_ids[0]))

        bbox_lbp_reference, overflow_bbbox = crop_image(init_image, self.init_gt_bbox[0], self.init_gt_bbox[1],
                                                        self.init_gt_bbox[2], self.init_gt_bbox[3])
        distance_lbp_list = []
        object_in_video = True
        lbp_threshold = 100
        lbp_threshold_factor = 3.0  # threshold for lbp distance =lbp_threshold_factor* lbp_threshold
        lbp_threshold_factor_nonfound = 4.0  # threshold for lbp distance =lbp_threshold_factor* lbp_threshold

        config = InferenceConfig()
        model = modellib.MaskRCNN(mode=ARG_MODE, model_dir=ARG_COCO_MODEL_PATH, config=config)
        # Loading weights
        model.load_weights(ARG_COCO_MODEL_PATH, by_name=True)

        if self.auxiliary_tracker_type == "KCF":
            self.auxiliary_tracker.init(init_image, self.init_gt_bbox)
        else:
            self.auxiliary_tracker.init(init_image, np.array(self.init_gt_bbox, dtype=np.float64))

            # to cumpute correlation
            self.kcf = KCF(padding=1, features='color', interp_factor=0.00)
            self.kcf.init(init_image, self.init_gt_bbox)


            # self.auxilary_tracker.init(init_image,self.init_gt_bbox)

            # starting to track
        for frame_index in range(1, self.nb_frames):
            print("frame index: ", frame_index)

            image_name = self.image_ids[frame_index]

            # Why this condition is here ???
            if (image_name[-4:] == ".jpg"):

                image = skimage.io.imread(os.path.join(self.img_list_dir, image_name))

                b_box_image = np.copy(image)

                dims = image.shape
                canvas_shape_x_y_format = [dims[1], dims[0]]

                results = model.detect([image], verbose=1, particles=particle_mask_normalize(self.particles[0],
                                                                                             os.path.join(
                                                                                                 self.img_list_dir,
                                                                                                 os.listdir(
                                                                                                     self.img_list_dir)[
                                                                                                     0]), config)[0])

                bbox_tdot, tdot_cond_satisfied = self.choose_best_bbox_tdot_correlation(image, results[0])

                if tdot_cond_satisfied == True:
                    self.auxiliary_tracker.init(image, bbox_tdot)

                    self.frame_tracker.append("TDOT")
                    self.bbox_results[frame_index - 1] = bbox_tdot
                    # set particle seed
                    bbox_for_particle_seed = bbox_tdot

                    b_box_image = cv2.rectangle(image, (bbox_tdot[0], bbox_tdot[1]),
                                                (bbox_tdot[2] + bbox_tdot[0], bbox_tdot[3] + bbox_tdot[1]), (0, 0, 255),
                                                4)

                    # crop bbox
                    x, y, w, h = bbox_tdot
                    bbox_lbp_candidate, overflow_bbbox = crop_image(image, x, y, w, h)

                    # resize bbo to reference bounding box
                    bbox_lbp_candidate_resize = resize_bbox(bbox_lbp_candidate, bbox_lbp_reference.shape[1],
                                                            bbox_lbp_reference.shape[0])

                    if bbox_lbp_candidate is not None:
                        lbp_hist_reference, bins = lbp_decimal_8(bbox_lbp_reference, hist_conc=True)
                        lbp_hist_condidate, bins = lbp_decimal_8(bbox_lbp_candidate_resize, hist_conc=True)

                        dist_chi_square_hist = chi_square_dist(lbp_hist_reference, lbp_hist_condidate)
                    else:
                        dist_chi_square_hist = -1.0

                    if dist_chi_square_hist < lbp_threshold_factor * lbp_threshold:
                        # CASE = 1
                        distance_lbp_list.append(dist_chi_square_hist)
                        lbp_threshold = sum(distance_lbp_list) / float(len(distance_lbp_list))

                    else:
                        # CASE = 2
                        # WHAT ABOUT THRESHOLD VALUE- lbp_threshold ???????
                        distance_lbp_list.clear()

                    bbox_lbp_reference, overflow_bbbox = crop_image(image, x, y, w, h)
                    object_in_video = True


                elif tdot_cond_satisfied == False and object_in_video == True:

                    bbox_corr = self.auxiliary_tracker.update(image)
                    #                    self.bbox_results[frame_index-1] = bbox_corr
                    #                    bbox_for_particle_seed = bbox_corr

                    #
                    #                    response_corr_pvpr = self.auxilary_tracker.response_p_arr_pvpr
                    #                    response_corr_psr = self.auxilary_tracker.response_p_arr_psr
                    #                    response_corr_epsr = self.auxilary_tracker.response_p_arr_epsr

                    bbox_corr_ = np.array(bbox_corr, dtype=np.int32)
                    x_final_corr, y_final_corr, w_final_corr, h_final_corr = bbox_corr_[0], bbox_corr_[1], bbox_corr_[
                        2], bbox_corr_[3]

                    overflow = are_corners_inimage(image, x_final_corr, y_final_corr, w_final_corr, h_final_corr)

                    if overflow != None:
                        #                        CASE = 3
                        #    object not found - give coordinates to -1.0
                        object_in_video == False
                        # match_bbox_for_particle_seed = [0.0, 0.0, 0.0, 0.0]
                        bbox_for_particle_seed = bbox_corr
                        self.bbox_results[frame_index - 1] = np.array([0, 0, 0, 0])
                        dist_chi_square_hist = -1.0
                        self.frame_tracker.append("NONE")
                    else:

                        bbox_lbp_candidate, overflow_bbbox = crop_image(image, x_final_corr, y_final_corr, w_final_corr,
                                                                        h_final_corr)

                        bbox_lbp_candidate_resize = resize_bbox(bbox_lbp_candidate, bbox_lbp_reference.shape[1],
                                                                bbox_lbp_reference.shape[0])

                        lbp_hist_reference, bins = lbp_decimal_8(bbox_lbp_reference, hist_conc=True)
                        lbp_hist_condidate, bins = lbp_decimal_8(bbox_lbp_candidate_resize, hist_conc=True)

                        dist_chi_square_hist = chi_square_dist(lbp_hist_reference, lbp_hist_condidate)

                        if dist_chi_square_hist < lbp_threshold_factor_nonfound * lbp_threshold:  # Test case 1
                            #                         if True: #Test case 2
                            #                            CASE = 4
                            self.bbox_results[frame_index - 1] = bbox_corr_
                            distance_lbp_list.append(dist_chi_square_hist)
                            lbp_threshold = sum(distance_lbp_list) / float(len(distance_lbp_list))
                            bbox_for_particle_seed = bbox_corr
                            object_in_video = True
                            b_box_image = cv2.rectangle(image, (bbox_corr_[0], bbox_corr_[1]),
                                                        (bbox_corr_[2] + bbox_corr_[0], bbox_corr_[3] + bbox_corr_[1]),
                                                        (0, 0, 255), 4)
                            self.frame_tracker.append("KCF")
                        else:
                            #                            CASE = 5
                            distance_lbp_list.clear()
                            object_in_video = False
                            self.bbox_results[frame_index - 1] = np.array([0, 0, 0, 0])
                            bbox_for_particle_seed = [0.0, 0.0, 0.0, 0.0]
                            self.frame_tracker.append("NONE")


                elif tdot_cond_satisfied == False and object_in_video == False:

                    self.bbox_results[frame_index - 1] = np.array([0, 0, 0, 0])
                    bbox_for_particle_seed = [0, 0, 0, 0]
                    self.frame_tracker.append("NONE")

                self.sample_particles(bbox_for_particle_seed, canvas_shape_x_y_format)
                print("\n./output_img/" + self.image_ids[frame_index] + "\n")
                cv2.imwrite("./output_img_kcf_kcf/" + self.image_ids[frame_index], b_box_image)

    def short_term_tracking(self):

        config = InferenceConfig()
        model = modellib.MaskRCNN(mode=ARG_MODE, model_dir=ARG_COCO_MODEL_PATH, config=config)
        # Loading weights
        model.load_weights(ARG_COCO_MODEL_PATH, by_name=True)

        init_image = skimage.io.imread(os.path.join(self.img_list_dir, self.image_ids[0]))

        if self.auxiliary_tracker_type == "KCF":
            self.auxiliary_tracker.init(init_image, self.init_gt_bbox)
        else:
            self.auxiliary_tracker.init(init_image, np.array(self.init_gt_bbox, dtype=np.float64))

            # to cumpute correlation
            self.kcf = KCF(padding=1, features='color', interp_factor=0.00)
            self.kcf.init(init_image, self.init_gt_bbox)

        b_box_image = cv2.rectangle(init_image, (self.init_gt_bbox[0], self.init_gt_bbox[1]), (
        self.init_gt_bbox[2] + self.init_gt_bbox[0], self.init_gt_bbox[3] + self.init_gt_bbox[1]), (255, 255, 255), 4)
        cv2.imwrite("./output_img_kcf_kcf/" + self.image_ids[0], b_box_image)

        # starting to track
        for frame_index in range(1, self.nb_frames):
            print("frame index: ", frame_index)

            image_name = self.image_ids[frame_index]

            # Why this condition is here ???
            if (image_name[-4:] == ".jpg"):

                image = skimage.io.imread(os.path.join(self.img_list_dir, image_name))

                dims = image.shape
                canvas_shape_x_y_format = [dims[1], dims[0]]

                results = model.detect([image], verbose=1, particles=particle_mask_normalize(self.particles[0],
                                                                                             os.path.join(
                                                                                                 self.img_list_dir,
                                                                                                 os.listdir(
                                                                                                     self.img_list_dir)[
                                                                                                     0]), config)[0])

                # results of the first sample on mini-batch
                r = results[0]

                #                bbox_tdot, tdot_cond_satisfied = self.choose_best_bbox_tdot(image,r)

                #                print("\n\n\nresults\n\n")
                #                print(r['rois'])
                #                print("\n\n")

                bbox_tdot, tdot_cond_satisfied = self.choose_best_bbox_tdot_correlation(image, r)

                if tdot_cond_satisfied == True:
                    #                    x,y,w,h = bbox_tdot

                    if self.auxiliary_tracker_type == "KCF":
                        # initialize correlation filter
                        self.auxiliary_tracker.init(image, bbox_tdot)
                    else:
                        x, y, w, h = bbox_tdot
                        self.auxiliary_tracker.re_init_params(np.array([y + h / 2, x + w / 2], dtype=np.float32),
                                                              np.array([h, w], dtype=np.float32))
                        self.kcf.init(image, bbox_tdot)

                    b_box_image = cv2.rectangle(image, (bbox_tdot[0], bbox_tdot[1]),
                                                (bbox_tdot[2] + bbox_tdot[0], bbox_tdot[3] + bbox_tdot[1]), (0, 0, 255),
                                                4)

                    self.bbox_results[frame_index - 1] = bbox_tdot
                    self.frame_tracker.append("TDOT")
                    # set particle seed
                    bbox_for_particle_seed = bbox_tdot
                # bbox_for_particle_seed = [x,y,w,h]

                else:
                    #                    CORRELATION

                    if self.auxiliary_tracker_type == "KCF":
                        bbox_corr = self.auxiliary_tracker.update(image)
                        self.bbox_results[frame_index - 1] = bbox_corr
                        self.frame_tracker.append("KCF")
                        bbox_for_particle_seed = bbox_corr

                        bbox_corr_ = np.array(bbox_corr, dtype=np.int32)
                        b_box_image = cv2.rectangle(image, (bbox_corr_[0], bbox_corr_[1]),
                                                    (bbox_corr_[2] + bbox_corr_[0], bbox_corr_[3] + bbox_corr_[1]),
                                                    (0, 255, 0), 4)
                    else:
                        bbox_siamfc = self.auxiliary_tracker.update(image)
                        bbox_siamfc = bbox_siamfc.astype(np.int32)
                        self.bbox_results[frame_index - 1] = bbox_siamfc
                        self.frame_tracker.append("SIAM")
                        bbox_for_particle_seed = bbox_siamfc

                        b_box_image = cv2.rectangle(image, (bbox_siamfc[0], bbox_siamfc[1]),
                                                    (bbox_siamfc[2] + bbox_siamfc[0], bbox_siamfc[3] + bbox_siamfc[1]),
                                                    (0, 255, 0), 4)

                self.sample_particles(bbox_for_particle_seed, canvas_shape_x_y_format)

                print("\n./output_img/" + self.image_ids[frame_index] + "\n")
                cv2.imwrite("./output_img_kcf_kcf/" + self.image_ids[frame_index], b_box_image)

                #                print("\nbbox: ",self.bbox_results[frame_index-1])
                #                cv2.imshow('image',b_box_image)
                #                cv2.waitKey(0)

                #            cv2.destroyAllWindows()

    def choose_best_bbox_tdot(self, current_image, result_model):

        total_nb_candidate = result_model['class_ids'].size
        tdot_candidate_satisfied = False
        bbox_final_tdot = None
        max_class_score = None

        for score_id in range(total_nb_candidate):

            y1, x1, y2, x2 = result_model['rois'][score_id]
            obj_score = result_model['scores'][score_id]
            predicted_class_id = result_model['class_ids'][score_id]
            #            predicted_class = class_names[predicted_class_id]

            x_up_left, y_up_left, width, height, score_part = self.coco_to_voc_bbox_converter(y1, x1, y2, x2, obj_score)
            bbox_candidate = [x_up_left, y_up_left, width, height]

            target_condition_satisfied = False
            if predicted_class_id == self.target_id:
                target_condition_satisfied = True

            if max_class_score is None and target_condition_satisfied == True:

                max_class_score = obj_score
                tdot_candidate_satisfied = True
                bbox_final_tdot = bbox_candidate

            elif max_class_score is not None and obj_score > max_class_score and target_condition_satisfied == True:

                max_class_score = obj_score[0]
                tdot_candidate_satisfied = True
                bbox_final_tdot = bbox_candidate

                #            print(obj_score, target_condition_satisfied)
                #            b_box_image = cv2.rectangle(current_image, (bbox_candidate[0], bbox_candidate[1]), (bbox_candidate[2]+bbox_candidate[0], bbox_candidate[3]+bbox_candidate[1]), (0,255,0), 4)
                #
                #            cv2.imshow('image',b_box_image)
                #            cv2.waitKey(0)
                #            cv2.destroyAllWindows()

        return bbox_final_tdot, tdot_candidate_satisfied

    def choose_best_bbox_tdot_correlation(self, current_image, result_model):

        total_nb_candidate = result_model['class_ids'].size
        print("\ntotal_nb_candidate: ", total_nb_candidate)
        print()
        max_corr_score = None
        tdot_candidate_satisfied = False
        bbox_final_tdot = None
        for score_id in range(total_nb_candidate):

            y1, x1, y2, x2 = result_model['rois'][score_id]
            obj_score = result_model['scores'][score_id]
            predicted_class_id = result_model['class_ids'][score_id]
            # predicted_class = class_names[predicted_class_id]
            x_up_left, y_up_left, width, height, score_part = self.coco_to_voc_bbox_converter(y1, x1, y2, x2, obj_score)

            bbox_candidate = [x_up_left, y_up_left, width, height]

            if self.auxiliary_tracker_type == "KCF":
                _ = self.auxiliary_tracker.compute_correlation(current_image, np.array([bbox_candidate]))

                response_p_arr_psr = self.auxiliary_tracker.response_p_arr_psr
                # response_p_arr_pvpr = self.auxilary_tracker.response_p_arr_pvpr
                # response_p_arr_epsr = self.auxilary_tracker.response_p_arr_epsr
            else:
                _ = self.kcf.compute_correlation(current_image, np.array([bbox_candidate]))

                response_p_arr_psr = self.kcf.response_p_arr_psr
                # response_p_arr_pvpr = self.auxilary_tracker.response_p_arr_pvpr
                # response_p_arr_epsr = self.auxilary_tracker.response_p_arr_epsr

            #            print("response_p_arr_psr: ", response_p_arr_psr[0], obj_score)
            #            b_box_image = cv2.rectangle(current_image, (bbox_candidate[0], bbox_candidate[1]), (bbox_candidate[2]+bbox_candidate[0], bbox_candidate[3]+bbox_candidate[1]), (0,255,0), 4)
            #
            #            cv2.imshow('image',b_box_image)
            #            cv2.waitKey(0)
            #            cv2.destroyAllWindows()

            target_condition_satisfied = False
            if predicted_class_id == self.target_id:
                target_condition_satisfied = True

            if max_corr_score is None and target_condition_satisfied == True:

                max_corr_score = response_p_arr_psr[0]
                tdot_candidate_satisfied = True
                bbox_final_tdot = bbox_candidate

            elif max_corr_score is not None and response_p_arr_psr[0] > max_corr_score and target_condition_satisfied:

                max_corr_score = response_p_arr_psr[0]
                tdot_candidate_satisfied = True
                bbox_final_tdot = bbox_candidate

        return bbox_final_tdot, tdot_candidate_satisfied

    def sample_particles(self, bbox_particle_seed, canvas_shape_x_y_format):
        random.seed(5)
        self.particles = utils.sample_bbox(bbox_particle_seed, stdev_coord=GAUSS_STDEV_COORD,
                                           stdev_scale=GAUSS_STDEV_SCALE, count=PARTICLE_CNT,
                                           canvas_shape=canvas_shape_x_y_format)

    def find_target_id(self):
        target_id = -1
        for i, class_name in enumerate(class_names):
            if class_name == self.object_name:
                target_id = i
                break

        return target_id

    def coco_to_voc_bbox_converter(self, y1, x1, y2, x2, roi_score=-1):
        w = x2 - x1
        h = y2 - y1
        return x1, y1, w, h, roi_score

    def read_image_file_names(self, img_list_dir):
        image_ids = os.listdir(img_list_dir)
        sorted_image_ids = sorted(image_ids, key=lambda x: x[:-4])
        sorted_image_ids = list(filter(lambda x: x[-4:] == ".jpg", sorted_image_ids))

        return sorted_image_ids

    def convert_coordinates(self, coord_list):
        #        (1)----(2)
        #        \       \
        #        \       \
        #        (4)----(3)
        #
        #        From format
        #            x1,y1,x2,y2,x3,y3,x4,y4
        #        to format
        #            x1,y1,w,h
        #        :return:

        top_left_x = float(coord_list[0])
        top_left_y = float(coord_list[1])
        top_right_x = float(coord_list[2])
        top_right_y = float(coord_list[3])
        bottom_right_x = float(coord_list[4])
        bottom_right_y = float(coord_list[5])
        bottom_left_x = float(coord_list[6])
        bottom_left_y = float(coord_list[7])

        left_avg = math.floor((top_left_x + bottom_left_x) / 2)
        top_avg = math.floor((top_left_y + top_right_y) / 2)
        width_avg = math.floor(((top_right_x - top_left_x) + (bottom_right_x - bottom_left_x)) / 2)
        height_avg = math.floor(((bottom_left_y - top_left_y) + (bottom_right_y - top_right_y)) / 2)
        return [left_avg, top_avg, width_avg, height_avg]

# ----------------------------------------------------------------------------

# img_dir_list = ["data/ball1_img/",
#                "data/ball2_img/",
#                "data/basketball_img/",
#                "data/birds1_img/",
#                "data/car1_img/",
#                "data/nature_img/",
#                "data/wiper_img/"]
#
# video_name = ["ball1",
#            "ball2",
#            "basketball",
#            "birds1",
#            "car1",
#            "nature",
#            "wiper"]
#
# video_gt = [[496, 419, 40, 42],
#          [532, 53, 13, 13],
#          [190, 210, 35, 107],
#          [508, 233, 83, 18],
#          [243, 165, 116, 111],
#          [749, 245, 165, 79],
#          [287, 253, 41, 41]]
#
# video_obj = ["sports ball",
#           "sports ball",
#           "person",
#           "bird",
#           "car",
#           "bird",
#           "car"]
#
# for i, video_name in enumerate(video_name,0):
#
#    if i>3:
#
#        print(i, video_name, video_obj[i], video_gt[i])
#
#        tr_kcf = Tracker(img_list_dir= img_dir_list[i],
#                     init_gt_bbox = video_gt[i],
#                     object_name= video_obj[i],
#                     output_dir ="",
#                     auxilary_tracker_type = "KCF",
#                     short_term = True)
#
#        a_kcf = tr_kcf.bbox_results
#
#        with open( video_name+"_kcf.txt", 'a+') as f:
#            for j in range(a_kcf.shape[0]):
#                res_bbox_frame = "{}\t{}\t{}\t{}\n".format(format(a_kcf[j,0],'.4f'), format(a_kcf[j,1],'.4f'), format(a_kcf[j,2],'.4f'), format(a_kcf[j,3],'.4f'))
#
#                f.write(res_bbox_frame)
#
#        tr_siam = Tracker(img_list_dir= img_dir_list[i],
#                     init_gt_bbox = video_gt[i],
#                     object_name= video_obj[i],
#                     output_dir ="",
#                     auxilary_tracker_type = "SIAM",
#                     short_term = True)
#
#        a_siam = tr_siam.bbox_results
#
#        with open( video_name+"_siam.txt", 'a+') as f:
#            for j in range(a_kcf.shape[0]):
#                res_bbox_frame = "{}\t{}\t{}\t{}\n".format(format(a_siam[j,0],'.4f'), format(a_siam[j,1],'.4f'), format(a_siam[j,2],'.4f'), format(a_siam[j,3],'.4f'))
#                f.write(res_bbox_frame)
# ------------------------------------------------------------------------



