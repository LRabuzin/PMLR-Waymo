import tensorflow.compat.v1 as tf
import numpy as np
import MinkowskiEngine as ME

tf.enable_eager_execution()

from waymo_open_dataset.utils import frame_utils

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

def extract_learnables(frame):
    parsed_frame = frame_utils.parse_range_image_and_camera_projection(frame)

    points, _ = frame_utils.convert_range_image_to_point_cloud(frame, parsed_frame[0], parsed_frame[1], parsed_frame[3], keep_polar_features=True)
    points_ri2, _ = frame_utils.convert_range_image_to_point_cloud(frame, parsed_frame[0], parsed_frame[1], parsed_frame[3], ri_index=1, keep_polar_features=True)
    points = np.concatenate(points)
    points_ri2 = np.concatenate(points_ri2)

    features = np.concatenate((points[...,:3], points_ri2[...,:3]))
    coordinates = np.concatenate((points[...,3:], points_ri2[...,3:]))

    point_labels = convert_range_image_to_point_cloud_labels(frame, parsed_frame[0], parsed_frame[2])
    point_labels_ri2 = convert_range_image_to_point_cloud_labels(frame, parsed_frame[0], parsed_frame[2], ri_index=1)
    point_labels = np.concatenate(point_labels)
    point_labels_ri2 = np.concatenate(point_labels_ri2)

    labels = np.concatenate((point_labels, point_labels_ri2))

    return coordinates, features, labels