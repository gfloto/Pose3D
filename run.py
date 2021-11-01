import os
import time
import numpy as np
import cv2
import acapture

import util
import nn1
import nn2

import detectron2
from common.custom_dataset import CustomDataset

window_size = 250 # number of images in temporal convolution

path = 'out_files' # save path

# get nets and dataset
net1, net2 = util.get_nets()

# init camera (s)
print('get camera')
cap = acapture.open(0)

# code to remove all old run files
files = os.listdir(path)
for f in files:
    os.remove(os.path.join(path, f))

frame_count = 0
while True:
    # get image, run though first neural network
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    nn1.run(net1, img, path, frame_count, window_size)

    # process results to prepare for second neural network
    util.process(path) 

    # build dataloader object
    if frame_count == 0:
        dataset = CustomDataset(os.path.join(path, 'out 2.npz'))

    # run data through second neural network
    nn2.run(net2, path, dataset)
    frame_count += 1   
