import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def run(net1, img, path, frame_count, window=100):
    boxes, boxes2 = [], []
    segments, segments2 = [], []
    keypoints, keypoints2 = [], []
    
    # push image through network  
    outputs = net1(img)['instances'].to('cpu')

    # process
    bbox_tensor = outputs.pred_boxes.tensor.numpy()
    scores = outputs.scores.numpy()[:, None]
    bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
    
    kps = outputs.pred_keypoints.numpy()
    kps_xy = kps[:, :, :2]
    kps_prob = kps[:, :, 2:3]
    kps_logit = np.zeros_like(kps_prob)
    kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
    kps = kps.transpose(0, 2, 1)

    boxes.append(bbox_tensor)
    segments.append(None)
    keypoints.append(kps)

    # include previous data for temporal convolutions
    if frame_count > 0:
        data = np.load(os.path.join(path, 'out 1.npz'), allow_pickle=True)

        read_boxes = data['boxes']
        read_keypoints = data['keypoints']

        for i in range(read_boxes.shape[0]):
            boxes.append(read_boxes[i])

        for i in range(read_keypoints.shape[0]):
            keypoints.append(read_keypoints[i])

    if len(boxes) > window:
        boxes = boxes[:-1]
        keypoints = keypoints[:-1]

    # Video resolution
    metadata = {
        'w': img.shape[1],
        'h': img.shape[0],
    }

    np.savez_compressed(os.path.join(path, 'out 1.npz'), boxes=boxes, segments=segments, keypoints=keypoints, metadata=metadata)
