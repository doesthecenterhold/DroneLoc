import numpy as np
from numpy.linalg import norm
import cv2
import matplotlib.pyplot as plt
import sys
# from scipy.spatial.transform import Rotation as R
sys.path.append('../')

from DroneLoc.datasets.GES_dataset import GES_dataset
from DroneLoc.datasets.VPAir_dataset import VPAir_dataset
from DroneLoc.utils.image_trans import center_max_crop, posrot_to_transform, invert_transform, trans_path_to_xy
from DroneLoc.utils.math import isRotationMatrix, rotationMatrixToEulerAngles # type: ignore
from DroneLoc.utils.math import unit_vector, angle_vector

dataset = GES_dataset()
K = dataset.K

image_step = 1
# Initiate SIFT detector
sift = cv2.SIFT_create()

# Parameters for FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

MIN_MATCH_COUNT = 15

# Assumption that the camera is pointing down
# Ignoring the geometry of objects, the ideal normal of the plane
# on which the keypoints lie, is given bellow.
ideal_normal = np.array([0, 0, 1])

cur_transform = np.eye(4)
cur_est_transform = np.eye(4)

gt_transforms = [cur_transform.copy()]
est_transforms = [cur_est_transform.copy()]

# for n in range(image_step, len(dataset), image_step):
for n in range(image_step, 200, image_step):

    # Get the correct indexes of the imagest
    img_first = dataset[n-image_step]
    img_second = dataset[n]

    # Load the images as grayscale images
    print(img_first['path'])
    first = cv2.imread(img_first['path'], cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(img_second['path'], cv2.IMREAD_GRAYSCALE)

    # first = center_max_crop(first)
    # second = center_max_crop(second)

    # Load the ground truth locations
    xgt1, ygt1, zgt1 = img_first['position']
    xanglegt1, yanglegt1, zanglegt1 = img_first['rotation']
    # T1 = posrot_to_transform((img_first['position']),(img_first['rotation']), Rot=img_first['R'])
    T1 = posrot_to_transform((img_first['position']),(img_first['rotation']))

    print('Gt1', T1)

    xgt2, ygt2, zgt2 = img_second['position']
    xanglegt2, yanglegt2, zanglegt2 = img_second['rotation']
    # T2 = posrot_to_transform((img_second['position']),(img_second['rotation']), Rot=img_second['R'])
    T2 = posrot_to_transform((img_second['position']),(img_second['rotation']))


    T12 = invert_transform(T1) @ T2

    cur_transform = cur_transform @ T12
    gt_transforms.append(cur_transform.copy())

    gt_dist = np.linalg.norm(T12[:3,3])

    # Prints to verify basic facts about transforms
    # print(np.linalg.norm( [xgt2-xgt1, ygt2-ygt1, zgt2-zgt1]))
    # print ('Transform matrix length', np.linalg.norm(T12[:3,3]))
    # print('T1 plus the transform ', T1 @ T12)
    # print('T2', T2)
    
    # Find keypoints and calculate descriptors
    kp1, ds1 = sift.detectAndCompute(first, None)
    kp2, ds2 = sift.detectAndCompute(second, None)

    # Find matches based on descriptors
    matches = flann.knnMatch(ds1, ds2, k=2)
    # matches = flann.match(ds1, ds2)

    # matches_s = sorted(matches, key = lambda x:x.distance)[:100]

    # Need to draw only good matches, so create a mask
    matchesMask = []
    good_matches = []

    # Ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
            matchesMask.append([1,0])

    print(f'Found {len(good_matches)} good matches per Lowe\'s paper.')
    good_matches_s = sorted(good_matches, key = lambda x:x.distance)[:1000]

    # draw_params = dict(matchColor = (0,255,0),
    #                    singlePointColor = (255,0,0),
    #                    matchesMask = matchesMask,
    #                    flags = cv2.DrawMatchesFlags_DEFAULT)

    if len(good_matches)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # print('Homography is :', H)

        # num, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)


        # best_angle = np.inf
        # bR, bt = 0, 0
        # for Rm,tu,nu in zip(Rs, ts, ns):
        #     R = rotationMatrixToEulerAngles(Rm)
        #     n = nu.squeeze()
        #     t = tu.squeeze()
        #     angle = angle_vector(n, ideal_normal)
        #     if angle<best_angle:
        #         best_angle = angle
        #         bR = R
        #         bt = t
        #     print('Rot', R, 'trans', t, 'normal', n, angle)

        # print('Most likely solution is:', bR, bt)

        E, mask = cv2.findEssentialMat(dst_pts, src_pts, cameraMatrix=K)
        _1, R, t, mask = cv2.recoverPose(E, dst_pts, src_pts, K)
        t = t.squeeze()
        EST12 = np.eye(4)
        EST12[:3,:3] = R
        EST12[:3, 3] = gt_dist*t/np.linalg.norm(t)

        # EST12 = posrot_to_transform(bt, bR)

        cur_est_transform = cur_est_transform @ EST12
        est_transforms.append(cur_est_transform.copy())

        print('Gt trans', T12)
        print('Estimated trans', EST12)

    else:
        print('Not enough good matches, skipping frames!')
    
    # result_img = cv2.drawMatchesKnn(first,kp1,second,kp2,matches,None)
    # result_img = cv2.drawMatches(first,kp1,second,kp2,good_matches_s,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    
    # plt.imshow(result_img)
    # plt.show()

gt = trans_path_to_xy(gt_transforms)
est = trans_path_to_xy(est_transforms)

plt.plot(gt[0], gt[1], c = 'b')
plt.plot(est[0], est[1], c = 'r')
plt.show()