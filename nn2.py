import os
import time
import numpy as np
from common.camera import *
from common.generators import UnchunkedGenerator

def fetch(dataset, subjects, keypoints, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = 1
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]


    return out_camera_params, out_poses_3d, out_poses_2d

def evaluate(net2, test_generator, joints_left, joints_right, action=None, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        net2.eval()

        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            t = time.time()
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = net2(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                if not use_trajectory_model:
                    predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            return predicted_3d_pos.squeeze(0).cpu().numpy()


def run(net2, path, dataset):
    s_info = 'out 1'
    causal_shift = 0

    receptive_field = net2.receptive_field()
    pad = (receptive_field - 1) // 2 # Padding on each side

    keypoints = np.load(os.path.join(path, 'out 2.npz'), allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    subjects_test = [s_info]
    action_filter = None

    # fetching operation
    cameras_valid, poses_valid, poses_valid_2d = fetch(dataset, subjects_test, keypoints, action_filter)

    # make generator
    test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    input_keypoints = keypoints[s_info]['custom'][0].copy()
    ground_truth = None

    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=True,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    prediction = evaluate(net2, gen, joints_left, joints_right)

    # Invert camera transformation
    cam = dataset.cameras()[s_info][0]
    # If the ground truth is not available, take the camera extrinsic params from a random subject.
    # They are almost the same, and anyway, we only need this for visualization purposes.
    for subject in dataset.cameras():
        if 'orientation' in dataset.cameras()[subject][0]:
            rot = dataset.cameras()[subject][0]['orientation']
            break
    prediction = camera_to_world(prediction, R=rot, t=0)

    # We don't have the trajectory, but at least we can rebase the height
    prediction = np.expand_dims(prediction[0], axis=0)

    np.save(os.path.join(path, 'joints.npy'), prediction)
    np.save(os.path.join(path, 'cam_az'), np.array(cam['azimuth']))

