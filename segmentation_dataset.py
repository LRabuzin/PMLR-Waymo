import os
import torch
from tqdm import TqdmExperimentalWarning
import tensorflow.compat.v1 as tf
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

tf.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


class WaymoSegmentationDataset(Dataset):

    def __init__(self, root_dir, mode='training', quantization_size = 0.2, transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.quantization_size = quantization_size
        self.transform = transform
        self.input_files = [os.join(root_dir, mode, f) for f in os.listdir(os.join(root_dir, mode)) if os.isfile(os.join(root_dir, mode, f))]
        self.curr_i = 0
        self.data = []
    
    def __len__(self):
        if self.mode == 'training':
            return 23691
        elif self.mode == 'validation':
            return 5976
        else:
            return "FIND TESTING SIZE"
    
    def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
        """Convert segmentation labels from range images to point clouds.

        Args:
            frame: open dataset frame
            range_images: A dict of {laser_name, [range_image_first_return,
            range_image_second_return]}.
            segmentation_labels: A dict of {laser_name, [range_image_first_return,
            range_image_second_return]}.
            ri_index: 0 for the first return, 1 for the second return.

        Returns:
            point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
            points that are not labeled.
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        point_labels = []
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            range_image_mask = range_image_tensor[..., 0] > 0

            if c.name in segmentation_labels:
                sl = segmentation_labels[c.name][ri_index]
                sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
                sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
            else:
                num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
                sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)
            
            point_labels.append(sl_points_tensor.numpy()[...,1])
        return point_labels
    
    def __getitem__(self, idx):
        if self.data:
            return self.data.pop()
        
        curr_scene = tf.data.TFRecordDataset(self.input_files[self.curr_i], compression_type='')
        self.curr_i = (self.curr_i+1)%len(self.input_files)

        curr_frames = []
        for data in curr_scene:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if frame.lasers[0].ri_return1.segmentation_label_compressed:
                curr_frames.append(frame)
        
        # (range_images, camera_projections, segmentation_labels, range_image_top_pose)
        parsed_frames = [frame_utils.parse_range_image_and_camera_projection(frame) for frame in curr_frames]

        for parsed_frame, frame in zip(parsed_frames, curr_frames):
            points, _ = frame_utils.convert_range_image_to_point_cloud(frame, parsed_frame[0], parsed_frame[1], parsed_frame[3], keep_polar_features=True)
            points_ri2, _ = frame_utils.convert_range_image_to_point_cloud(frame, parsed_frame[0], parsed_frame[1], parsed_frame[3], ri_index=1, keep_polar_features=True)
            points = np.concatenate(points)
            points_ri2 = np.concatenate(points_ri2)

            inner_features = []
            inner_coordinates = []
            inner_labels = []
            for point in points:
                inner_features.append(point[:3])
                inner_coordinates.append(point[3:])

            for point in points_ri2:
                inner_features.append(point[:3])
                inner_coordinates.append(point[3:])

            point_labels = self.convert_range_image_to_point_cloud_labels(frame, parsed_frame[0], parsed_frame[2])
            point_labels_ri2 = self.convert_range_image_to_point_cloud_labels(frame, parsed_frame[0], parsed_frame[2], ri_index=1)
            point_labels = np.concatenate(point_labels)
            point_labels_ri2 = np.concatenate(point_labels_ri2)

            for label in point_labels:
                inner_labels.append(label)
            
            for label in point_labels_ri2:
                inner_labels.append(label)

            inner_features = np.asarray(inner_features)
            inner_coordinates = np.asarray(inner_coordinates)
            inner_labels = np.asarray(inner_labels)

            discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
                coordinates=inner_coordinates,
                features=inner_features,
                labels=inner_labels,
                quantization_size=self.quantization_size,
                ignore_label=0)

            self.data.append((discrete_coords, unique_feats, unique_labels))
        
        return self.data.pop()

if __name__=="__main__":
    dataset = WaymoSegmentationDataset(root_dir='/cluster/scratch/lrabuzin/waymo_data_updated')
    print(dataset[0])