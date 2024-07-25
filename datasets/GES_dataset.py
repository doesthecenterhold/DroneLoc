import numpy as np
import json
import pyproj
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('../')

from DroneLoc.utils.image_trans import center_max_crop, posrot_to_transform, invert_transform, trans_path_to_xy


np.set_printoptions(precision=4)

class GES_dataset:
    def __init__(self, path="/home/matej/Datasets/DroneLoc/Test1_Ljubljana_150m_80fov_90deg"):
        
        # Load the folders
        self.path = Path(path)
        self.image_folder = None
        self.anno_json = None
        for ppath in self.path.iterdir():
            if ppath.is_dir():
                self.image_folder = ppath
            elif ppath.suffix == ".json":
                self.anno_json = ppath
        
        assert self.image_folder is not None
        assert self.anno_json is not None

        with open(str(self.anno_json), 'r') as f:
            self.anno = json.load(f)

        # Coordinate conversion
        self.lla_to_ecef_trans = pyproj.Transformer.from_crs({"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
                                                             {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'})

        self.ecef_to_lla_trans = pyproj.Transformer.from_crs({"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
                                                             {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'})

        # Variables to hold the dataset
        self.image_prefix = self.anno_json.stem
        self.num_images = self.anno['numFrames']
        self.width = self.anno['width']
        self.height = self.anno['height']
        self.fov = self.anno['cameraFrames'][0]['fovVertical']

        self.images = self.load_dataset()

        h = 1080
        y_fov = np.deg2rad(80)

        cx = cy = h//2
        fy = h/(2*np.tan(y_fov/2))
        fx = fy

        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
        
        self.K = K

        print('The dataset has been loaded!')

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        return self.images[i]

    def load_dataset(self):
        
        images = []
        for n in range(self.num_images+1):
            img = {'path':str(self.image_folder / f"{self.image_prefix}_{n:04}.jpeg"),
                   'position':(self.anno['cameraFrames'][n]['position']['x'],
                               self.anno['cameraFrames'][n]['position']['y'],
                               self.anno['cameraFrames'][n]['position']['z']),
                   'rotation':(self.anno['cameraFrames'][n]['rotation']['x'],
                               self.anno['cameraFrames'][n]['rotation']['y'],
                               self.anno['cameraFrames'][n]['rotation']['z']),
                   'coordinates':(self.anno['cameraFrames'][n]['coordinate']['latitude'],
                                  self.anno['cameraFrames'][n]['coordinate']['longitude'],
                                  self.anno['cameraFrames'][n]['coordinate']['altitude'])}
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
    
    def trans_to_gt0(self):

        x, y, z = self.images[0]['position']
        rx, ry, rz = self.images[0]['rotation']
        GT0 = posrot_to_transform((x,y,z),(rx,ry,rz))
        GTfirst = GT0
        floating_transform = np.eye(4)
        images_norm = [floating_transform]

        unrot = invert_transform(GTfirst)

        for n in range(1, len(self.images)):
            img = self.images[n]
            x, y, z = img['position']
            rx, ry, rz = img['rotation']

            GT1 = posrot_to_transform((x,y,z),(rx,ry,rz))
            delta_gt = invert_transform(GT0) @ GT1

            # gt_test = unrot @ GT1

            floating_transform = floating_transform @ delta_gt

            print('relative position of next camera frame')
            print(delta_gt)

            # return

            images_norm.append(floating_transform)

            GT0 = GT1

        return images_norm, GT0

if __name__ == "__main__":

    ds = GES_dataset()
    anno = ds.images[0]

    transforms, gt0 = ds.trans_to_gt0()

    xs = [c[0,3] for c in transforms]
    ys = [c[1,3] for c in transforms]
    zs = [c[2,3] for c in transforms]

    plt.plot(xs, ys)
    plt.show()

    plt.plot(zs)
    plt.show()

    # lats = [x['coordinates'][0] for x in ds.images]
    # lons = [x['coordinates'][1] for x in ds.images]
    # alts = [x['coordinates'][2] for x in ds.images]
    # plt.plot(lons, lats)
    # plt.plot(alts)
    # plt.show()

    # lat, lon, alt = anno['coordinates']
    # x, y, z = anno['position']
    # print('Original lat, lon, alt', anno['coordinates'])
    # print('Original x, y, z', anno['position'])

    # print('Converted x, y, z', ds.lla_to_ecef(lat, lon, alt))
    # print('Converted lat, lon, alt', ds.ecef_to_lla(x, y, z))