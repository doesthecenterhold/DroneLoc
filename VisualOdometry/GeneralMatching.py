import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
import pickle
sys.path.append('../')

from DroneLoc.datasets.GES_dataset import GES_dataset
from DroneLoc.utils.image_trans import posrot_to_transform, invert_transform, trans_path_to_xy


HWK = (480, 640)
HWI = (1080, 1920)

def scale_keypoints(keypoints, hw_kpts=HWK, hw_imgs=HWI):
    ratios = [hw_imgs[1]/hw_kpts[1], hw_imgs[0]/hw_kpts[0]]
    scaled_keypoints = keypoints * ratios
    return scaled_keypoints

def draw_keypoints(image, keypoints):
    for keypoint in keypoints:   
        cv2.circle(image, np.int32(keypoint), 5, color=(0,255,0), thickness=-1)

    return image

def draw_matches(joined_image, keys1, keys2, howmany=-1):
    # if howmany!=-1:
    #     rnd_inds = np.random.randint(low=0, high=len(keys1), size=howmany)
    #     keys1=keys1[rnd_inds]
    #     keys2=keys2[rnd_inds]
    fix = np.array([HWI[1], 0])
    for k1, k2 in zip(keys1, keys2):
        cv2.line(joined_image, np.int32(k1), np.int32(k2) + fix, color=(0,0,255), thickness=1)

    return joined_image

# Load dataset
image_step = 1
dataset = GES_dataset('/home/matej/Datasets/DroneLoc/Test1_Ljubljana_150m_80fov_90deg_3000')

# Load results
# src_pts = pickle.load('/home/matej/Programs/DroneLoc/data/Test1_Ljubljana_150m_80fov_90deg_3000/src_points.pkl')

src_pts = np.load('/home/matej/Programs/DroneLoc/data/Test1_Ljubljana_150m_80fov_90deg_3000/src_points.pkl', allow_pickle=True)
dst_pts = np.load('/home/matej/Programs/DroneLoc/data/Test1_Ljubljana_150m_80fov_90deg_3000/dst_points.pkl', allow_pickle=True)

K = np.array([[643.97, 0, 960], [0, 643.97, 540], [0, 0, 1]])


GT0 = posrot_to_transform((dataset[0]['position']),(dataset[0]['rotation']))
GT0 = np.eye(4)
OdomT = GT0

GT_matrices = [GT0]
T_matrices = [OdomT]
total_length = [0]

for n in range(image_step, 10, image_step):

    # Load consecutive images
    img_first = dataset[n-image_step]
    img_second = dataset[n]

    # Load GT transformation matrices
    GT0 = posrot_to_transform((img_first['position']),(img_first['rotation']))
    GT1 = posrot_to_transform((img_second['position']),(img_second['rotation']))

    # Get GT relative transform
    GT01 = invert_transform(GT0) @ GT1
    gt_length = np.linalg.norm(GT01[:3,3])
    total_length.append(total_length[-1] + gt_length)

    GT_matrices.append(GT_matrices[-1] @ GT01)

    print('GT relative')
    print(GT01)

    # Load appropriate matched keypoints
    keys_first = src_pts[n-image_step]
    keys_second = dst_pts[n-image_step]

    ### Visualize Keypoints
    # Load actual images
    first = cv2.imread(img_first['path']) 
    second = cv2.imread(img_second['path'])

    # Convert from BGR to RGB to plot with plt  
    first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    second = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)

    # Scale keypoints to original resolution
    keys_first = scale_keypoints(keys_first)
    keys_second = scale_keypoints(keys_second)

    first = draw_keypoints(first, keys_first)
    second = draw_keypoints(second, keys_second)

    joined_images = np.concatenate([first, second], axis=1)
    joined_images = draw_matches(joined_images, keys_first, keys_second)

    # E, inliers = cv2.findEssentialMat(keys_second, keys_first,K,method=cv2.RANSAC,
    #                                   prob=0.99999,threshold=1.0,maxIters=10000)
    
    E, inliers = cv2.findEssentialMat(keys_second, keys_first,K,method=cv2.USAC_MAGSAC, threshold=0.3)
    


    num_inliers, R, T, mask_pose = cv2.recoverPose(E, keys_second[inliers.ravel() == 1], keys_first[inliers.ravel() == 1], K)

    # Input GT translation length
    T = T.squeeze()
    T = T/np.linalg.norm(T) * gt_length

    T01 = np.eye(4)
    T01[:3, :3] = R
    T01[:3, 3] = T

    print('Estimated')
    print(T01)

    print('Essential matrix extimated with inliers ratio', np.sum(inliers)/len(inliers))
    plt.imshow(joined_images)
    plt.show()

    OdomT = OdomT @ T01

    T_matrices.append(OdomT)

gtxs, gtys, gtzs = trans_path_to_xy(GT_matrices)
txs, tys, tzs = trans_path_to_xy(T_matrices)

gtlats, gtlons, gtalts = dataset.ecef_to_lla(gtxs, gtys, gtzs)
tlats, tlons, talts = dataset.ecef_to_lla(txs, tys, tzs)

fig, axs = plt.subplots(2)
fig.suptitle('SuperPoint + SuperGlue Essential Ljubljana 3000')
axs[0].plot(gtlats, gtlons, color='b', label='GT')
axs[0].plot(tlats, tlons, color='r', label='SP+SG')
axs[1].plot(total_length, gtalts, color='b')
axs[1].plot(total_length, talts, color='r')

plt.show()