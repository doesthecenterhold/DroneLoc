import numpy as np
import json
import pyproj
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
from scipy.spatial.transform import Rotation as R
import pymap3d as pm
from pymap3d.ellipsoid import Ellipsoid
import shutil
sys.path.append('../')

from DroneLoc.utils.image_trans import center_max_crop, posrot_to_transform, invert_transform, trans_path_to_xy


np.set_printoptions(precision=4)

class GES_dataset:
    def __init__(self, path="/home/matej/Datasets/DroneLoc/Train10_Venice_150m_80fov_90deg_3000"):
        
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
        
        list = []
        images = []
        prev_position = np.array([0,0,0])
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
            
            dist = np.linalg.norm(prev_position - np.array(img['position']))
                                  
            if dist < 70:
                continue
            else:

            #     print('GOOD DISTANCE', img['path'])
            #     shutil.copyfile(img['path'], img['path'].replace('footage','footage_50m'))
                prev_position = np.array(img['position'])


            # else:
            #     print('BAD DISTANCE', img['path'])

            
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
            lat, lon, alt = img['coordinates']

            GT1 = posrot_to_transform((x,y,z),(rx,ry,rz))
            delta_gt = invert_transform(GT0) @ GT1

            # gt_test = unrot @ GT1

            floating_transform = floating_transform @ delta_gt

            print('Distance to next camera frame')
            print(np.linalg.norm(delta_gt[:3,3]), alt)

            # return

            images_norm.append(floating_transform)

            GT0 = GT1

        return images_norm, GT0
    
    def geodetic2ecef(self, lat, lon, h):

        lat = np.radians(lat)
        lon = np.radians(lon)

        # WGS-84 ellipsoid parameters
        a = 6378137.0              # Semi-major axis (m)
        f = 1 / 298.257223563      # Flattening
        e2 = f * (2 - f)           # Eccentricity squared
        
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)
        
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        X = (N + h) * cos_lat * cos_lon
        Y = (N + h) * cos_lat * sin_lon
        Z = (N * (1 - e2) + h) * sin_lat

        return X, Y, Z
    
    def ecef2ned_euler_angles(self, ax, ay, az, lat, lon):
        """
        Convert Euler angles from ECEF to NED frame.
        :param ecef_euler: [roll, pitch, yaw] in radians (ECEF frame)
        :param lat: Geodetic latitude in radians
        :param lon: Longitude in radians
        :return: [roll, pitch, yaw] in radians (NED frame)
        """

        ecef_euler = [ax, ay, az]

        # Step 1: Convert ECEF Euler angles to rotation matrix
        r_ecef = R.from_euler('xyz', ecef_euler, degrees=True).as_matrix()
        
        # Step 2: Compute ECEF-to-NED rotation matrix
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)
        
        R_ecef2ned = np.array([
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [-sin_lon,            cos_lon,          0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
        ])
        
        # Step 3: Rotate the ECEF rotation matrix to NED frame
        r_ned = R_ecef2ned @ r_ecef
        
        # Step 4: Convert NED rotation matrix back to Euler angles
        ned_euler = R.from_matrix(r_ned).as_euler('xyz', degrees=True)

        return ned_euler
    

def ecef_to_ned_rotation_matrix(lat, lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R_NE = np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon),  np.cos(lat)],
        [-np.sin(lon),               np.cos(lon),                0],
        [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]
    ])
    return R_NE

def euler_to_rotation_matrix(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,               cp * cr]
    ])
    return R

def rotation_matrix_to_euler(R):
    pitch = -np.arcsin(R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw

# def ecef2ned_orient(roll_ecef, pitch_ecef, yaw_ecef, lat, lon):
#     # Given Euler angles in ECEF
#     roll_ecef, pitch_ecef, yaw_ecef = np.deg2rad([roll_ecef, pitch_ecef, yaw_ecef])

#     # Convert to rotation matrix in ECEF
#     R_ecef = euler_to_rotation_matrix(roll_ecef, pitch_ecef, yaw_ecef)

#     # Rotation matrix from ECEF to NED
#     R_ecef_to_ned = ecef_to_ned_rotation_matrix(lat, lon)

#     # Transform the rotation matrix from ECEF to NED
#     R_ned = R_ecef_to_ned @ R_ecef @ R_ecef_to_ned.T

#     # Extract Euler angles in NED
#     roll_ned, pitch_ned, yaw_ned = rotation_matrix_to_euler(R_ned)

#     return np.rad2deg(roll_ned), np.rad2deg(pitch_ned), np.rad2deg(yaw_ned)

    # print("Roll (NED):", np.rad2deg(roll_ned))
    # print("Pitch (NED):", np.rad2deg(pitch_ned))
    # print("Yaw (NED):", np.rad2deg(yaw_ned))


models = {
    # Earth ellipsoids
    "maupertuis": {"name": "Maupertuis (1738)", "a": 6397300.0, "b": 6363806.283},
    "plessis": {"name": "Plessis (1817)", "a": 6376523.0, "b": 6355862.9333},
    "everest1830": {"name": "Everest (1830)", "a": 6377299.365, "b": 6356098.359},
    "everest1830m": {
        "name": "Everest 1830 Modified (1967)",
        "a": 6377304.063,
        "b": 6356103.039,
    },
    "everest1967": {
        "name": "Everest 1830 (1967 Definition)",
        "a": 6377298.556,
        "b": 6356097.55,
    },
    "airy": {"name": "Airy (1830)", "a": 6377563.396, "b": 6356256.909},
    "bessel": {"name": "Bessel (1841)", "a": 6377397.155, "b": 6356078.963},
    "clarke1866": {"name": "Clarke (1866)", "a": 6378206.4, "b": 6356583.8},
    "clarke1878": {"name": "Clarke (1878)", "a": 6378190.0, "b": 6356456.0},
    "clarke1860": {"name": "Clarke (1880)", "a": 6378249.145, "b": 6356514.87},
    "helmert": {"name": "Helmert (1906)", "a": 6378200.0, "b": 6356818.17},
    "hayford": {"name": "Hayford (1910)", "a": 6378388.0, "b": 6356911.946},
    "international1924": {"name": "International (1924)", "a": 6378388.0, "b": 6356911.946},
    "krassovsky1940": {"name": "Krassovsky (1940)", "a": 6378245.0, "b": 6356863.019},
    "wgs66": {"name": "WGS66 (1966)", "a": 6378145.0, "b": 6356759.769},
    "australian": {"name": "Australian National (1966)", "a": 6378160.0, "b": 6356774.719},
    "international1967": {
        "name": "New International (1967)",
        "a": 6378157.5,
        "b": 6356772.2,
    },
    "grs67": {"name": "GRS-67 (1967)", "a": 6378160.0, "b": 6356774.516},
    "sa1969": {"name": "South American (1969)", "a": 6378160.0, "b": 6356774.719},
    "wgs72": {"name": "WGS-72 (1972)", "a": 6378135.0, "b": 6356750.52001609},
    "grs80": {"name": "GRS-80 (1979)", "a": 6378137.0, "b": 6356752.31414036},
    "wgs84": {"name": "WGS-84 (1984)", "a": 6378137.0, "b": 6356752.31424518},
    "wgs84_mean": {"name": "WGS-84 (1984) Mean", "a": 6371008.7714, "b": 6371008.7714},
    "iers1989": {"name": "IERS (1989)", "a": 6378136.0, "b": 6356751.302},
    "pz90.11": {"name": "ПЗ-90 (2011)", "a": 6378136.0, "b": 6356751.3618},
    "iers2003": {"name": "IERS (2003)", "a": 6378136.6, "b": 6356751.9},
    "gsk2011": {"name": "ГСК (2011)", "a": 6378136.5, "b": 6356751.758},
    # Other worlds
    "mercury": {"name": "Mercury", "a": 2440500.0, "b": 2438300.0},
    "venus": {"name": "Venus", "a": 6051800.0, "b": 6051800.0},
    "moon": {"name": "Moon", "a": 1738100.0, "b": 1736000.0},
    "mars": {"name": "Mars", "a": 3396900.0, "b": 3376097.80585952},
    "jupyter": {"name": "Jupiter", "a": 71492000.0, "b": 66770054.3475922},
    "io": {"name": "Io", "a": 1829.7, "b": 1815.8},
    "saturn": {"name": "Saturn", "a": 60268000.0, "b": 54364301.5271271},
    "uranus": {"name": "Uranus", "a": 25559000.0, "b": 24973000.0},
    "neptune": {"name": "Neptune", "a": 24764000.0, "b": 24341000.0},
    "pluto": {"name": "Pluto", "a": 1188000.0, "b": 1188000.0},
}

if __name__ == "__main__":

    ds = GES_dataset()
    anno = ds.images[0]
    x,y,z = anno['position']
    lat, lon, alt = anno['coordinates']
    ax, ay, az = anno['rotation']

    elps_name = "wgs84_mean"
    ells = Ellipsoid.from_name(elps_name)

    # for ellps in models.keys():
    #     # print('### Model, ellps', ellps)
            
    #     ells = Ellipsoid.from_name(ellps)

    #     nlat, nlon, nalt= pm.ecef2geodetic(x,y,z, ells)
    #     nx, ny, nz  = pm.geodetic2ecef(lat, lon, alt, ells)

    #     print(ellps, abs(x-nx), abs(y-ny), abs(z - nz), abs(lat-nlat), abs(lon-nlon), abs(alt - nalt))

        # print('Error coords', abs(lat-nlat), abs(lon-nlon), abs(alt - nalt))

    # print('position error'. )

    print('Original lat, lon, alt', 0, 0, 0)
    # print('Converted lat, lon, alt', pm.ecef2geodetic(x,y,z, ells))

    print('Original x, y, z', anno['position'])
    print('Converted x, y, z', pm.geodetic2ecef(0, 0, 0, ells))
    # print('Custom converted x, y, z', ds.geodetic2ecef(lat, lon, alt))

    print('ECEF vector', 0,0,1)
    print('NED vector', pm.ecef2nedv(0, 0, 1, lat, lon))

    print('original orientation', ax, ay, az)
    print('NED orientation', ds.ecef2ned_euler_angles(ax, ay, az, lat, lon))
