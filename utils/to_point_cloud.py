import numpy as np
from PIL import Image
import imageio
import struct
import os

def get_pointcloud(color_image,depth_image,camera_intrinsics):
    """ creates 3D point cloud of rgb images by taking depth information
        input : color image: numpy array[h,w,c], dtype= uint8
                depth image: numpy array[h,w] values of all channels will be same
        output : camera_points, color_points - both of shape(no. of pixels, 3)
    """

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,image_width-1,image_width),
                                  np.linspace(0,image_height-1,image_height))
    camera_points_x = np.multiply(pixel_x-camera_intrinsics[0,2],depth_image/camera_intrinsics[0,0])
    camera_points_y = np.multiply(pixel_y-camera_intrinsics[1,2],depth_image/camera_intrinsics[1,1])
    camera_points_z = depth_image
    camera_points = np.array([camera_points_x,camera_points_y,camera_points_z]).transpose(1,2,0).reshape(-1,3)

    color_points = color_image.reshape(-1,3)

    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_image.flatten() > 0)[0]
    camera_points = camera_points[valid_depth_ind,:]
    color_points = color_points[valid_depth_ind,:]

    return camera_points,color_points

def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()

camera_intrinsics  = np.asarray([[7.067553100585937500e+02, 0, 5.456326819328060083e+02], \
    [0, 7.075133056640625000e+02, 3.899299663507044897e+02], [0, 0, 1]]) / 4.0
data_dirs = ['/home/x_keiik/154_ws/PDNE/experiments/PPFT_inference_2024-04-17-14:49:54/val/epoch-24/raw_data',
             '/home/x_keiik/154_ws/PDNE/experiments/PPFT_inference_2024-04-17-14:49:54/test/epoch-24/raw_data',
             '/home/x_keiik/154_ws/PDNE/experiments/CompletionFormer_inference_2024-04-17-15:00:33/val/epoch-38/raw_data',
             '/home/x_keiik/154_ws/PDNE/experiments/CompletionFormer_inference_2024-04-17-15:00:33/test/epoch-38/raw_data']

for data_dir in data_dirs:
    for data_sub_dir in os.listdir(data_dir):
        full_path = os.path.join(data_dir, data_sub_dir)
        gt_d = sparse_d = out_d = color_data = None
        for file in os.listdir(full_path):
            full_path_file = os.path.join(full_path, file)
            if 'gt_raw' in file:
                gt_d = imageio.imread(full_path_file)
            elif 'sparse_raw' in file:
                sparse_d = imageio.imread(full_path_file)
            elif 'out_raw' in file:
                out_d = imageio.imread(full_path_file)
            elif 'rgb' in file:
                color_data = imageio.imread(full_path_file)
        if not (gt_d is not None and sparse_d is not None and out_d is not None and color_data is not None):
            print("Missing data")
            continue
        
        file_names = ['gt', 'sparse', 'out']
        for i, depth_data in enumerate([gt_d, sparse_d, out_d]):
            output_filename = os.path.join(full_path, file_names[i] + ".ply")
            camera_points, color_points = get_pointcloud(color_data, depth_data, camera_intrinsics)
            write_pointcloud(output_filename, camera_points, color_points)
            print("Write {}".format(output_filename))