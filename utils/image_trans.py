import numpy as np
from scipy.spatial.transform import Rotation as R

def trans_path_to_xy(path):
    xs = [trans[0,3] for trans in path]
    ys = [trans[1,3] for trans in path]
    zs = [trans[2,3] for trans in path]

    return xs, ys, zs

def center_max_crop(image):
    h, w = image.shape
    hh, wh = int(h/2), int(w/2)
    mid_y, mid_x = int(h/2), int(w/2)
    cropped_image = image[mid_y-hh:mid_y+hh, mid_x-hh:mid_x+hh]
    return cropped_image

def posrot_to_transform(pos, rot, Rot=None):
    R1 = R.from_euler('XYZ', [rot], degrees=False)
    R1 = R1.as_matrix()

    T1 = np.eye(4)
    T1[:3, 3] = pos
    if Rot is None:
        T1[:3,:3] = R1
    else:
        R2 = np.eye(4)
        R2[:3,:3] = R1
        T1 = Rot @ T1
        # T1 = T1 @ R2

    return T1

def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]


    inverted_matrix = np.eye(4)
    inverted_matrix[:3, :3] = R.T
    inverted_matrix[:3, 3] = -R.T @ t

    return inverted_matrix
