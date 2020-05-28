from lxml import etree
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.signal import convolve2d
import cv2


def parse_cascade(xml):
    root = etree.fromstring(xml)

    cascade = root.find("cascade")
    width = int(cascade.find("width").text)
    height = int(cascade.find("height").text)
    features = cascade.find("features").getchildren()

    feature_matrices = np.zeros((len(features), height, width))
    for i, feature in enumerate(features):
        cur_matrix = np.zeros((height, width))
        for rect in feature.find("rects").getchildren():
            line = rect.text.strip().split(" ")
            x1, y1, x2, y2 = map(int, line[:4])
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            c = float(line[4])
            
            cur_matrix[y1:y2+1, x1:x2+1] = c
        
        feature_matrices[i] = cur_matrix

    stages = cascade.find("stages")
    stages_list = []
    for stage in stages.getchildren():
        if type(stage) == etree._Element:
            threshold = float(stage.find("stageThreshold").text)
            clfs = stage.find("weakClassifiers")
            
            classifiers = []
            for clf in clfs:
                internal_nodes = clf.find("internalNodes").text.strip().split(" ")
                feature_num = int(internal_nodes[2])
                feature_thresh = float(internal_nodes[3])
                
                leafs = clf.find("leafValues").text.strip().split(" ")
                less_leaf = float(leafs[0])
                greater_leaf = float(leafs[1])
                
                classifiers.append([feature_num, feature_thresh, less_leaf, greater_leaf])
            
            stages_list.append([threshold, classifiers])

    return stages_list, feature_matrices, width, height


def get_top_classifier_outputs(image, feature, thresh, less, greater, k):
    activation_map = convolve2d(image, feature, mode="valid")
    if greater > less:
        activation_map[activation_map < thresh] = 0
    else:
        activation_map[activation_map > thresh] = 0
    
    # top k non-zero activations
    flatten_activation_map = activation_map.flatten()
    top_indices = np.argpartition(flatten_activation_map, -k)[-k:]

    # filter zero activations
    top_indices = top_indices[flatten_activation_map[top_indices] > 0]
    
    return top_indices


def get_stage_images(image, stages, features, height, width, k):
    result_images = []

    for stage in stages:
        args = []
        for classifier in stage[1]:
            feature_num, thresh, less, greater = classifier
            arg = [image, features[feature_num], thresh, less, greater, k]
            args.append(arg)
        
        with Pool(cpu_count()) as pool:
            map_res = pool.starmap(get_top_classifier_outputs, args)

            white_area = np.zeros(image.shape, dtype=np.uint8)

            activation_shape = list(image.shape)
            activation_shape[0] -= height - 1
            activation_shape[1] -= width - 1

            for top_indices in map_res:
                for top_index in top_indices:
                    i, j = np.unravel_index(top_index, activation_shape)
                    white_area[i:i+height, j:j+width] = 255
            
            result_image = cv2.addWeighted(image, 0.5, white_area, 0.5, 1.0)
    
        result_images.append(result_image)
    return result_images


def save_video(video_name, stage_images, time_per_stage, frame_width, frame_height):
    fps = 10
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

    frames_per_stage = int(fps * time_per_stage)
    for image in stage_images:
        image_resized = cv2.resize(image, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        image_bgr = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)
        for i in range(frames_per_stage):
            out.write(image_bgr)

    out.release()
