import numpy as np
import pptk


def show_point_cloud(pc_data):
    # pc_data: point cloud data
    pptk.viewer(pc_data[:, :3])


if __name__ == '__main__':
    path_to_point_cloud = '/home/dtc/Data/KITTI/save/000000.bin'

    point_cloud_data = np.fromfile(path_to_point_cloud, '<f4')  # little-endian float32
    point_cloud_data = np.reshape(point_cloud_data, (-1, 4))  # x, y, z, r
    show_point_cloud(point_cloud_data)
