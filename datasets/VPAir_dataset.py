import numpy as np
import json
import csv
import cv2
import sys
import pyproj
import pymap3d as pm
import matplotlib.pyplot as plt
from pathlib import Path


sys.path.append('../')
from DroneLoc.utils.image_trans import center_max_crop, posrot_to_transform, invert_transform, trans_path_to_xy


def rotation_towards_center(x,y,z):
    v = np.array([-x, -y, -z])
    v = v / np.linalg.norm(v)
    vx, vy, vz = v

    k = np.array([0,0,1])

    # a = np.cross(k, v)

    cos_theta = np.dot(k, v)
    theta = np.arccos(cos_theta)

    I = np.eye(3)
    K = np.array([[0, 0, -vx],
                  [0, 0, -vy],
                  [vx, vy, 0]])
    
    R3 = I + np.sin(theta)*K + (1-cos_theta)*K@K
    R = np.eye(4)
    R[:3,:3] = R3

    return R


class VPAir_dataset:
    def __init__(self, path="/home/matej/Datasets/DroneLoc/VPAir"):
        
        # Load the folders
        self.path = Path(path)
        self.image_folder = None
        self.anno_json = None
        for ppath in self.path.iterdir():
            if ppath.is_dir() and not 'undistorted' in str(ppath):
                self.image_folder = ppath
            elif ppath.suffix == ".csv":
                self.anno_csv = ppath
        
        assert self.image_folder is not None
        assert self.anno_csv is not None

        with open(str(self.anno_csv), 'r') as f:
            lines = f.readlines()
            self.anno = []
            for line in lines[1:]:
                self.anno.append(line.split(','))

        # Coordinate conversion
        self.lla_to_ecef_trans = pyproj.Transformer.from_crs({"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
                                                             {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'})

        self.ecef_to_lla_trans = pyproj.Transformer.from_crs({"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
                                                             {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'})

        # Variables to hold the dataset
        self.image_prefix = ''
        self.num_images = len(self.anno)
        self.width = 800
        self.height = 600
        # self.fov = self.anno['cameraFrames'][0]['fovVertical']

        self.images = self.load_dataset()

        self.K = np.array([[750.62614972, 0, 402.41007535],
                           [0, 750.26301185, 292.98832147],
                           [0, 0, 1]])
        

        self.T_cam_imu = np.array([[0.012858067322034039, 0.9999102686262149, -0.0037582975656914553, -0.013410258642810948],
                                   [-0.9999169997865642, 0.012861034406219052, 0.0007663757808519512, 0.022027794132179236],
                                   [0.0008146426072014672, 0.003748131514807178, 0.9999926439067288, -0.2116875787833756],
                                   [0.0, 0.0, 0.0, 1.0]])

        self.distM = np.array([-0.11592226392258145, 0.1332261251415265, -0.00043977637330175616, 0.0002380609784102606])

        print('The dataset has been loaded!')


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        return self.images[i]

    def load_dataset(self):
        
        images = []
        for n in range(self.num_images):
            lat, lon, alt = list(map(float, (self.anno[n][1], self.anno[n][2], self.anno[n][3])))
            roll, pitch, yaw = list(map(float, (self.anno[n][4], self.anno[n][5], self.anno[n][6])))
            path = str(self.image_folder / f"{n+1:05}.png")

            x, y, z = self.lla_to_ecef(lat, lon, alt)
            R = rotation_towards_center(x, y, z)
            img = {'path':path,
                   'position':(x, y, z),
                   'rotation':(roll, pitch, yaw),
                   'coordinates':(lat, lon, alt),
                   'R':R}
            images.append(img)

        return images
    
    def get_vector_towards_earth_center(self, lat, lon, alt):
        alt2 = alt-1

        x1, y1, z1 = self.lla_to_ecef(lat, lon, alt)
        x2, y2, z2 = self.lla_to_ecef(lat, lon, alt2)

        x, y, z = x1-x2, y1-y2, z1-z2

        return x, y, z

    def lla_to_ecef(self, lat, lon, alt):
        #TODO fix function
        x, y, z = self.lla_to_ecef_trans.transform(lon, lat, alt, radians=False)
        return x, y, z

    def ecef_to_lla(self, x, y, z):
        #TODO fix function
        lon, lat, alt = self.ecef_to_lla_trans.transform(x, y, z, radians=False)
        return lat, lon, alt

if __name__ == "__main__":

    ds = VPAir_dataset()
    for img in ds.images:
        pt = img['path']
        img = cv2.imread(pt)
        newcameramatrix, _ = cv2.getOptimalNewCameraMatrix(ds.K, ds.distM, (800, 600), 1, (800, 600))
        new_img = cv2.undistort(img, ds.K, ds.distM, None, newcameramatrix)
        print(newcameramatrix)
        pt.replace('queries', 'queries_undistorted')
        print(pt)
        cv2.imwrite(pt, new_img)

    anno = ds.images[0]
    la0, lo0, alt0 = anno['coordinates']
    roll, pitch, yaw = anno['rotation']

    # no, ea, do = pm.geodetic2ned(la0, lo0, alt0, la0, lo0, alt0)
    # GT0 = posrot_to_transform((no, ea, do), (roll, pitch, yaw))
    rot90 = posrot_to_transform((0,0,0),(0,0,-np.pi/2))
    print(rot90)

    xs = []
    ys = []
    zs = []

    transforms = []

    for anno in ds.images:
        lat, lon, alt = anno['coordinates']
        roll, pitch, yaw = anno['rotation']
        no, ea, do = pm.geodetic2ned(lat, lon, alt, la0, lo0, alt0)

        # print(no, ea, do)
        GT_in0 = posrot_to_transform((no, ea, do), (roll, pitch, yaw))
        GT_in0 = ds.T_cam_imu @ GT_in0

        transforms.append(GT_in0)

        # print(GT_in0)

        xs.append(GT_in0[0,3])
        ys.append(GT_in0[1,3])
        zs.append(GT_in0[2,3])


    for n in range(1, len(transforms)):

        gt0 = transforms[n-1]
        gt1 = transforms[n]

        gt01 = invert_transform(gt0) @ gt1

        print(gt01)

    # plt.plot(xs, ys)
    # plt.show()
    # plt.plot(zs)
    # plt.show()
    
    lats = [x['coordinates'][0] for x in ds.images]
    lons = [x['coordinates'][1] for x in ds.images]
    alts = [x['coordinates'][2] for x in ds.images]

    # plt.plot(lons, lats)
    # plt.show()

    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(lons, lats)
    axs[1].plot(xs, ys)
    plt.show()


    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(alts)
    axs[1].plot(zs)
    plt.show()

    lat, lon, alt = anno['coordinates']
    x, y, z = anno['position']

    print('Original lat, lon, alt', anno['coordinates'])
    print('Original x, y, z', anno['position'])

    print('Converted x, y, z', ds.lla_to_ecef(lat, lon, alt))
    print('Converted lat, lon, alt', ds.ecef_to_lla(x, y, z))