import os
import numpy as np
from glob import glob
from common.data_utils import suggest_metadata

def decode(filename):
    # Latin1 encoding because Detectron runs on Python 2.7
    data = np.load(filename, allow_pickle=True)
    bb = data['boxes']
    kp = data['keypoints']
    metadata = data['metadata'].item()
    results_bb = []
    results_kp = []

    for i in range(len(bb)):
        if len(bb[i]) == 0 or len(kp[i]) == 0:
            # No bbox/keypoints detected for this frame -> will be interpolated
            results_bb.append(np.full(4, np.nan, dtype=np.float32)) # 4 bounding box coordinates
            results_kp.append(np.full((17, 4), np.nan, dtype=np.float32)) # 17 COCO keypoints
            continue
        best_match = np.argmax(bb[i][:, 4])
        best_bb = bb[i][best_match, :4]
        best_kp = kp[i][best_match].T.copy()
        results_bb.append(best_bb)
        results_kp.append(best_kp)

    bb = np.array(results_bb, dtype=np.float32)
    kp = np.array(results_kp, dtype=np.float32)
    kp = kp[:, :, :2] # Extract (x, y)

    print('{} frames processed'.format(len(bb)))

    return [{
        'start_frame': 0, # Inclusive
        'end_frame': len(kp), # Exclusive
        'bounding_boxes': bb,
        'keypoints': kp,
    }], metadata

def process(path):
    metadata = suggest_metadata('coco')
    metadata['video_metadata'] = {}

    output = {}
    files = glob(os.path.join(path, 'out 1.npz'))
    for i, f in enumerate(files):
        canonical_name = os.path.splitext(os.path.basename(f))[0]
        data, video_metadata = decode(f)
        output[canonical_name] = {}
        output[canonical_name]['custom'] = [data[0]['keypoints'].astype('float32')]

        metadata['video_metadata'][canonical_name] = video_metadata

    np.savez_compressed(os.path.join(path, 'out 2.npz'), positions_2d=output, metadata=metadata)

import torch
import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from common.model import *
from common.custom_dataset import CustomDataset
from detectron2.modeling import build_model

def get_nets():
    input_cfg = 'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
    model_path = 'model/pretrained_h36m_detectron_coco.bin'

    # build models
    print('building model 1')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(input_cfg))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(input_cfg)
    predictor = DefaultPredictor(cfg)

    print('building model 2')
    model_pos = TemporalModel(17, 2, 17, [3,3,3,3,3], causal=False,
                                dropout=0.25, channels=1024, dense=False)

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])
    model_traj = None

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
        print('using gpu')

    return predictor, model_pos
