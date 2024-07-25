import numpy as np
import json
import csv
import pyproj
import matplotlib.pyplot as plt
from pathlib import Path


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
            if ppath.is_dir():
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

    def lla_to_ecef(self, lat, lon, alt):
        #TODO fix function
        x, y, z = self.lla_to_ecef_trans.transform(lon, lat, alt, radians=False)
        return x, y, z

    def ecef_to_lla(self, x, y, z):
        #TODO fix function
        lon, lat, alt = self.ecef_to_lla_trans.transform(x, y, z, radians=False)
        return lat, lon, alt

# if __name__ == "__main__":

    # ds = GES_dataset()
    # anno = ds.images[0]

    # lats = [x['coordinates'][0] for x in ds.images]
    # lons = [x['coordinates'][1] for x in ds.images]

    # plt.plot(lons, lats)
    # plt.show()

    # lat, lon, alt = anno['coordinates']
    # x, y, z = anno['position']
    # print('Original lat, lon, alt', anno['coordinates'])
    # print('Original x, y, z', anno['position'])

    # print('Converted x, y, z', ds.lla_to_ecef(lat, lon, alt))
    # print('Converted lat, lon, alt', ds.ecef_to_lla(x, y, z))