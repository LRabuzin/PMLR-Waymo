import tensorflow.compat.v1 as tf
import numpy as np
import MinkowskiEngine as ME
import numpy as np
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset.protos import segmentation_submission_pb2
from waymo_open_dataset.utils import frame_utils,range_image_utils,transform_utils
from waymo_open_dataset import dataset_pb2

#-- new
import zlib

tf.enable_eager_execution()

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(__file__))

sys.path.append(os.path.join(parent_dir, 'dataset'))
sys.path.append(os.path.join(parent_dir, 'models'))

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_dataset
import minkunet
import argparse
from datetime import datetime



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



#-------- compressor copied from tutorial:

def compress_array(array: np.ndarray, is_int32: bool = False):
  """Compress a numpy array to ZLIP compressed serialized MatrixFloat/Int32.

  Args:
    array: A numpy array.
    is_int32: If true, use MatrixInt32, otherwise use MatrixFloat.

  Returns:
    The compressed bytes.
  """
  if is_int32:
    m = open_dataset.MatrixInt32()
  else:
    m = open_dataset.MatrixFloat()
  m.shape.dims.extend(list(array.shape))
  m.data.extend(array.reshape([-1]).tolist())
  return zlib.compress(m.SerializeToString())






def extract_learnables_modified(frame):
    parsed_frame = frame_utils.parse_range_image_and_camera_projection(frame)

    points, _ = frame_utils.convert_range_image_to_point_cloud(frame, parsed_frame[0], parsed_frame[1], parsed_frame[3], keep_polar_features=True)
    points_ri2, _ = frame_utils.convert_range_image_to_point_cloud(frame, parsed_frame[0], parsed_frame[1], parsed_frame[3], ri_index=1, keep_polar_features=True)

    labelled_len_1, labelled_len_2 = len(points[0]), len(points_ri2[0])

    points = np.concatenate(points)
    points_ri2 = np.concatenate(points_ri2)

    # indice meanings: [ number of labelled points 1, total point number from 1,
    #                     number of labelled points 2, total point number from 2 ]

    labelled_points_num = np.array([labelled_len_1, len(points), labelled_len_2, len(points_ri2)])


    features = np.concatenate((points[...,:3], points_ri2[...,:3]))
    coordinates = np.concatenate((points[...,3:], points_ri2[...,3:]))

    point_labels = convert_range_image_to_point_cloud_labels(frame, parsed_frame[0], parsed_frame[2])
    point_labels_ri2 = convert_range_image_to_point_cloud_labels(frame, parsed_frame[0], parsed_frame[2], ri_index=1)
    point_labels = np.concatenate(point_labels)
    point_labels_ri2 = np.concatenate(point_labels_ri2)

    labels = np.concatenate((point_labels, point_labels_ri2))

    return coordinates, features, labels, labelled_points_num






def semseg_for_one_frame(frame, model, device='cpu'):


            #---------------------------------------------------------------
             #---------------------------------------------------------------
              #---------------------------------------------------------------

            parsed_frame = frame_utils.parse_range_image_and_camera_projection(frame)
            #return range_images, camera_projections, seg_labels, range_image_top_pose

            points_ri1, cp_points_ri1 = frame_utils.convert_range_image_to_point_cloud(frame, parsed_frame[0], parsed_frame[1], parsed_frame[3], keep_polar_features=True)
            points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(frame, parsed_frame[0], parsed_frame[1], parsed_frame[3], ri_index=1, keep_polar_features=True)

            labelled_len_1, labelled_len_2 = len(points_ri1[0]), len(points_ri2[0])

            label_points_ri1 = points_ri1[0]
            label_points_ri2 = points_ri2[0]

            points_ri1 = np.concatenate(points_ri1)
            points_ri2 = np.concatenate(points_ri2)

            # indice meanings: [ number of labelled points 1, total point number from 1,
            #                     number of labelled points 2, total point number from 2 ]

            labelled_points_num = np.array([labelled_len_1, len(points_ri1), labelled_len_2, len(points_ri2)])


            features = np.concatenate((points_ri1[...,:3], points_ri2[...,:3]))
            coordinates = np.concatenate((points_ri1[...,3:], points_ri2[...,3:]))

            #features_1 = label_points_ri1[...,:3]
            #features_2 =  label_points_ri2[...,:3]
            #coordinates_1 =  label_points_ri1[...,3:]
            #coordinates_2 =  label_points_ri2[...,3:]

            point_labels_ri1 = convert_range_image_to_point_cloud_labels(frame, parsed_frame[0], parsed_frame[2])
            point_labels_ri2 = convert_range_image_to_point_cloud_labels(frame, parsed_frame[0], parsed_frame[2], ri_index=1)
            point_labels_ri1 = np.concatenate(point_labels_ri1)
            point_labels_ri2 = np.concatenate(point_labels_ri2)

            labels = np.concatenate((point_labels_ri1, point_labels_ri2))


#-----------------------------------------------------------------------------------------------------------------------------------
              #---------------------------------------------------------------
               #---------------------------------------------------------------

            # new modifications - for submission formatting
            context_name = frame.context.name
            frame_timestamp_micros = frame.timestamp_micros

            range_images = parsed_frame[0]

            #print(len(frame.context.laser_calibrations))

            laser_calibration_0 = sorted(frame.context.laser_calibrations, key=lambda c: c.name)[0] 
            range_image_1 = range_images[laser_calibration_0.name][0]
            range_image_2 = range_images[laser_calibration_0.name][1]
            #print(range_image_1.shape)
            #print(range_image_2.shape)


            laser_name_str = dataset_pb2.LaserName.Name.Name(laser_calibration_0.name)

            if len(laser_calibration_0.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test

                beam_inclinations_1 = range_image_utils.compute_inclination(
                tf.constant( [ laser_calibration_0.beam_inclination_min, laser_calibration_0.beam_inclination_max]),
                height=range_image_1.shape.dims[0])

                beam_inclinations_2 = range_image_utils.compute_inclination(
                tf.constant( [ laser_calibration_0.beam_inclination_min, laser_calibration_0.beam_inclination_max]),
                height=range_image_2.shape.dims[0])


            else:
                beam_inclinations_1 = tf.constant(laser_calibration_0.beam_inclinations)
                beam_inclinations_2 = tf.constant(laser_calibration_0.beam_inclinations)

            beam_inclinations_1 = tf.reverse(beam_inclinations_1, axis=[-1])
            beam_inclinations_2 = tf.reverse(beam_inclinations_2, axis=[-1])


            #extrinsic = laser_calibration_0.extrinsic
            print(beam_inclinations_1)
            extrinsic = np.reshape(np.array(laser_calibration_0.extrinsic.transform), [4, 4])

#--------------------------------------------------------------------------------------------------

            discrete_coords, unique_feats, unique_labels, index, inverse = ME.utils.sparse_quantize(
                coordinates=coordinates,
                features=features,
                labels=labels,
                quantization_size=0.2,
                ignore_label=0,
                return_index=True,
                return_inverse=True)

            #discrete_coords_1, unique_feats_1 = ME.utils.sparse_quantize(
            #    coordinates=coordinates_1,
            #    features=features_1,
            #    quantization_size=0.2,
            #    ignore_label=-100)

            #discrete_coords_2, unique_feats_2 = ME.utils.sparse_quantize(
            #    coordinates=coordinates_2,
            #    features=features_2,
            #    quantization_size=0.2,
            #    ignore_label=-100)
    
            coords, feats = ME.utils.sparse_collate([discrete_coords], [unique_feats])
            #coords_1, feats_1 = ME.utils.sparse_collate([discrete_coords_1], [unique_feats_1])
            #coords_2, feats_2 = ME.utils.sparse_collate([discrete_coords_2], [unique_feats_2]) 

            model_in = ME.SparseTensor( features=feats, 
                coordinates=coords, device = device)

            #model_in_1 = ME.SparseTensor( features=feats_1,
            #    coordinates=coords_1, device = device)

            #model_in_2 = ME.SparseTensor( features=feats_2,
            #    coordinates=coords_2, device = device)


            out = model(model_in) 
            out_squeezed = out.F.squeeze()
            out_coords = out.C.squeeze()
            


            #out_1 = model(model_in_1)
            #out_squeezed_1 = out_1.F.squeeze()
            #out_coords_1 = out_1.C.squeeze()


            #out_2 = model(model_in_2)
            #out_squeezed_2 = out_2.F.squeeze()
            #out_coords_2 = out_2.C.squeeze()


            #print("initial coords shape")
            #print(coordinates.shape)
            #print("out coords shape")
            #print(out_coords.shape)
            #print("Label output of model:")
            #print(out_squeezed.shape)

            TOP_LIDAR_ROW_NUM = 64
            TOP_LIDAR_COL_NUM = 2650
             
            origin_labels = out_squeezed.detach().numpy().argmax(axis=1)[inverse]

            top_lidar_points_ri1 = coordinates[:labelled_points_num[0]]
            top_lidar_labels_ri1 = origin_labels[:labelled_points_num[0]]

            top_lidar_points_ri2 = coordinates[labelled_points_num[1]:labelled_points_num[1]+labelled_points_num[2]]
            top_lidar_labels_ri2 = origin_labels[labelled_points_num[1]:labelled_points_num[1]+labelled_points_num[2]]

            labels_r1 = top_lidar_labels_ri1
            labels_r2 = top_lidar_labels_ri2
            #print("labels r1")
            #print(labels_r1.shape)
            #print("labels r2")
            #print(labels_r2.shape)
            
            #out_labels_1 = out_squeezed_2.detach().numpy().argmax(axis=1)
            #out_labels_2 = out_squeezed_2.detach().numpy().argmax(axis=1)

            


            range_image_1,ri_indices_1,ri_ranges_1 =range_image_utils.build_range_image_from_point_cloud(points_vehicle_frame = tf.expand_dims(top_lidar_points_ri1, axis=0),
                                       num_points = tf.convert_to_tensor(value=[len(top_lidar_points_ri1)]),
                                       extrinsic  = tf.expand_dims(extrinsic, axis=0),
                                       inclination= tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations_1), axis=0),
                                       range_image_size=[TOP_LIDAR_ROW_NUM,TOP_LIDAR_COL_NUM],
                                       point_features=None,
                                       dtype=tf.float32,
                                       scope=None)
            
            range_image_1 = range_image_1[0]
            ri_indices_1 = ri_indices_1[0]

            #print(top_lidar_points_ri2.shape)

            print(ri_indices_1)

            print(range_image_1)


            range_image_2,ri_indices_2,ri_ranges_2 =range_image_utils.build_range_image_from_point_cloud(points_vehicle_frame = tf.expand_dims(top_lidar_points_ri2, axis=0),
                                       num_points = tf.convert_to_tensor(value=[len(top_lidar_points_ri2)]),
                                       extrinsic  = tf.expand_dims(extrinsic, axis=0),
                                       inclination= tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations_2), axis=0),
                                       range_image_size=[TOP_LIDAR_ROW_NUM,TOP_LIDAR_COL_NUM],
                                       point_features=None,
                                       dtype=tf.float32,
                                       scope=None)

            range_image_2 = range_image_2[0]
            ri_indices_2 = ri_indices_2[0]

            #print("range image shape after")
            #print(range_image_2.shape)

            #print("indice shape after")
            #print(ri_indices_2.shape) 
            
            #print("ri indices sample:")
            #print(ri_indices_1[:,0])

            #print("labels shape")
            #print(labels_r1.shape)
            #print(labels_r1)


            # Assign the dummy class to all valid points (in the range image)
            range_image_pred = np.zeros((TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)

            range_image_pred[ ri_indices_1[:, 0], ri_indices_1[:, 1], 1] = labels_r1

            range_image_pred_ri2 = np.zeros( (TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)

            range_image_pred_ri2[ ri_indices_2[:, 0], ri_indices_2[:, 1], 1] = labels_r2


            # Construct the SegmentationFrame proto.
            segmentation_frame = segmentation_metrics_pb2.SegmentationFrame()

            segmentation_frame.context_name = context_name
            segmentation_frame.frame_timestamp_micros = frame_timestamp_micros

            laser_semseg = open_dataset.Laser()

            laser_semseg.name = open_dataset.LaserName.TOP
    
            laser_semseg.ri_return1.segmentation_label_compressed = compress_array( range_image_pred, is_int32=True)
            laser_semseg.ri_return2.segmentation_label_compressed = compress_array( range_image_pred_ri2, is_int32=True)

            segmentation_frame.segmentation_labels.append(laser_semseg)

            return segmentation_frame




def dataset_semseg(root_dir, output_dir, frame_info_path,
        model_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    testing_set_frame_file = frame_info_path

    context_name_timestamp_tuples = [x.rstrip().split(',') for x in (open(testing_set_frame_file, 'r').readlines())]
    #print(context_name_timestamp_tuples)
    segmentation_frame_list = segmentation_metrics_pb2.SegmentationFrameList()

    # ADD MODEL LOADING CODE
    model = minkunet.MinkUNet14A(in_channels=3, out_channels=23, D=3)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    input_files = [os.path.join(root_dir, "testing", f) for f in os.listdir(os.path.join(root_dir, "testing")) if os.path.isfile(os.path.join(root_dir, "testing", f))]

    for idx, raw_scene in enumerate(input_files):
        if idx % 10 == 0:
            print('Processing %d/%d run segments...' % (idx, len(input_files)))

        scene = tf.data.TFRecordDataset(raw_scene, compression_type='')
        for data in scene:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            context_name = frame.context.name
            timestamp = frame.timestamp_micros

            #print(context_name + " "+ str(timestamp) )

            if [context_name, str(timestamp)] in context_name_timestamp_tuples:

                segmentation_frame = semseg_for_one_frame(frame=frame, model=model, device=device)

                segmentation_frame_list.frames.append(segmentation_frame)

    print('Total number of frames: ', len(segmentation_frame_list.frames))


    # Create the submission file, which can be uploaded to the eval server.
    submission = segmentation_submission_pb2.SemanticSegmentationSubmission()
    submission.account_name = 'lovro.rabuzin@gmail.com'
    submission.unique_method_name = 'Minkowski_1'
    submission.affiliation = 'ETH Zurich'
    submission.authors.append('Lovro Rabuzin')
    submission.authors.append('Mert Ertugrul')
    submission.authors.append('Anton Alexandrov')

    submission.description = "A sparse convolution based U-Net approach - aka MinkuNet"
    submission.method_link = 'NA'
    submission.sensor_type = 1
    submission.number_past_frames_exclude_current = 0
    submission.number_future_frames_exclude_current = 0
    submission.inference_results.CopyFrom(segmentation_frame_list)

    output_filename = os.path.join(output_dir, 'wod_semseg_test_set_minkunet_submission_1.bin')

    f = open(output_filename, 'wb')
    f.write(submission.SerializeToString())
    f.close()




if __name__ == '__main__':

    dataset_semseg('/cluster/scratch/mertugrul/waymo_data_updated', '/cluster/home/mertugrul/PMLR-Waymo', frame_info_path='/cluster/home/mertugrul/PMLR-Waymo/dataset/3d_semseg_test_set_frames.txt',
                   model_path="/cluster/home/mertugrul/PMLR-Waymo/checkpoint.pth")
